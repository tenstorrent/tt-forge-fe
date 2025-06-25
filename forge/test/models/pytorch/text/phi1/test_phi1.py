# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoTokenizer,
    PhiForCausalLM,
    PhiForSequenceClassification,
    PhiForTokenClassification,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model

variants = ["microsoft/phi-1", "microsoft/phi-1_5"]

class PhiModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            past_key_values_length = 0
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
            )
            logits = self.model(attention_mask=causal_attention_mask, inputs_embeds=inputs_embeds).logits
        else:
            logits = self.model(input_ids=input_ids).logits
        return logits


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi1_causal_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1_5,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load tokenizer and model from HuggingFace
    model = download_model(PhiForCausalLM.from_pretrained, variant, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token 

    framework_model = PhiModelWrapper(model)
    framework_model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True,
    )

    sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs, module_name)

    # Model Verification
    verify(sample_inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi1_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1_5,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(PhiForTokenClassification.from_pretrained, variant, use_cache=False)

    framework_model = PhiModelWrapper(model)
    framework_model.eval()

    # input_prompt
    input_prompt = "HuggingFace is a company based in Paris and New York"
    inputs = tokenizer(input_prompt, return_tensors="pt")
    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_token_class_ids = co_out[0].argmax(-1)[0]
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids]

    print(f"Context: {input_prompt}")
    print(f"Answer: {predicted_tokens_classes}")


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi1_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI1_5,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load tokenizer and model from HuggingFace
    model = download_model(PhiForSequenceClassification.from_pretrained, variant, use_cache=False)
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)

    framework_model = PhiModelWrapper(model)
    framework_model.eval()

    # input_prompt
    input_prompt = "the movie was great!"
    inputs = tokenizer(input_prompt, return_tensors="pt")

    inputs = [inputs["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, inputs, module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_value = co_out[0].argmax(-1).item()
    print(f"Predicted Sentiment: {model.config.id2label[predicted_value]}")
