# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from test.utils import download_model


# Adding Wrapper to pass position ids to avoid dynamic shape errors
class RobertaWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        token_type_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        return self.model(input_ids, attention_mask, token_type_ids, position_ids)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["xlm-roberta-base"])
def test_roberta_masked_lm(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ROBERTA,
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModelForMaskedLM.from_pretrained, variant, return_dict=False)
    framework_model = RobertaWrapper(framework_model)
    framework_model.eval()

    # Input processing
    text = "Hello I'm a <mask> model."
    input_tokens = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(
        inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))
    )

    # post processing
    logits = co_out[0]
    mask_token_index = (input_tokens["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["cardiffnlp/twitter-roberta-base-sentiment"])
def test_roberta_sentiment_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.ROBERTA,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load Bart tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AutoModelForSequenceClassification.from_pretrained, variant, return_dict=False)
    framework_model = RobertaWrapper(framework_model)

    # Example from multi-nli validation set
    text = """Great road trip views! @ Shartlesville, Pennsylvania"""

    # Data preprocessing
    input_tokens = tokenizer.encode(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = [input_tokens]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_value = co_out[0].argmax(-1).item()
    print(f"Predicted Sentiment: {framework_model.config.id2label[predicted_value]}")
