# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.utils import download_model
import forge
import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from test.models.utils import build_module_name, Framework


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_roberta_masked_lm(record_forge_property):
    variant = "xlm-roberta-base"

    module_name = build_module_name(framework=Framework.PYTORCH, model="roberta", variant=variant, task="mlm")

    record_forge_property("module_name", module_name)

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(AutoModelForMaskedLM.from_pretrained, variant)

    # Input processing
    text = "Hello I'm a <mask> model."
    input_tokens = tokenizer.encode(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = torch.zeros_like(input_tokens)
    attention_mask[input_tokens != 1] = 1

    inputs = [input_tokens, attention_mask]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_roberta_sentiment_pytorch(record_forge_property):
    variant = "cardiffnlp/twitter-roberta-base-sentiment"

    module_name = build_module_name(framework=Framework.PYTORCH, model="roberta", variant=variant, task="sentiment")

    record_forge_property("module_name", module_name)

    # Load Bart tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(AutoModelForSequenceClassification.from_pretrained, variant)

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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
