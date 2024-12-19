# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.utils import download_model
import forge
import pytest
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_roberta_masked_lm(test_device):
    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, "xlm-roberta-base")
    model = download_model(AutoModelForMaskedLM.from_pretrained, "xlm-roberta-base")

    compiler_cfg = forge.config._get_compiler_config()  # load compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_roberta_masked_lm", compiler_cfg=compiler_cfg)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_roberta_sentiment_pytorch(test_device):
    # Load Bart tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, "cardiffnlp/twitter-roberta-base-sentiment")
    model = download_model(
        AutoModelForSequenceClassification.from_pretrained, "cardiffnlp/twitter-roberta-base-sentiment"
    )

    compiler_cfg = forge.config._get_compiler_config()  # load compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_roberta_sentiment", compiler_cfg=compiler_cfg)
