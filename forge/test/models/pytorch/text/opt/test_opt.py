# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoTokenizer,
    OPTForCausalLM,
    OPTForQuestionAnswering,
    OPTForSequenceClassification,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.utils import download_model


class OptModelWrapper(torch.nn.Module):
    def __init__(self, model, text_embedding):
        super().__init__()
        self.model = model
        self.text_embedding = text_embedding

    def forward(self, input_ids, attention_mask):
        inputs_embeds = self.text_embedding(input_ids)
        past_key_values_length = 0
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
        )
        position_ids = torch.cumsum(attention_mask, dim=1)
        position_ids = (position_ids * attention_mask - 1).long()
        position_ids = position_ids[:, past_key_values_length:]
        outputs = self.model(
            attention_mask=causal_attention_mask, inputs_embeds=inputs_embeds, position_ids=position_ids
        )
        if isinstance(outputs, (CausalLMOutputWithPast, SequenceClassifierOutputWithPast)):
            return outputs.logits
        elif isinstance(outputs, QuestionAnsweringModelOutput):
            return outputs.start_logits, outputs.end_logits
        else:
            return outputs


variants = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    pytest.param(
        "facebook/opt-1.3b",
        marks=[
            pytest.mark.xfail(
                reason="Data mismatch between framework and compiled model output. Issue Link: https://github.com/tenstorrent/tt-mlir/issues/4174"
            )
        ],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_opt_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.OPT,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    model = download_model(OPTForCausalLM.from_pretrained, variant, use_cache=False)
    framework_model = OptModelWrapper(model=model, text_embedding=model.model.decoder.embed_tokens)
    framework_model.eval()

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = "My name is Thomas and my main"
    input_tokens = tokenizer(
        prefix_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_opt_qa(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.OPT, variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(OPTForQuestionAnswering.from_pretrained, variant, use_cache=False)
    framework_model = OptModelWrapper(model=model, text_embedding=model.model.decoder.embed_tokens)
    framework_model.eval()

    # Load data sample
    question, context = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    pcc = 0.99
    if variant in ["facebook/opt-125m", "facebook/opt-1.3b"]:
        pcc = 0.95

    # Model Verification
    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_opt_sequence_classification(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.OPT,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(OPTForSequenceClassification.from_pretrained, variant, use_cache=False)
    framework_model = OptModelWrapper(model=model, text_embedding=model.model.decoder.embed_tokens)
    framework_model.eval()

    # Load data sample
    review = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
    )

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_value = co_out[0].argmax(-1).item()
    print(f"Predicted Sentiment: {model.config.id2label[predicted_value]}")
