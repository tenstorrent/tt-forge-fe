# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.utils import download_model
import forge
import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from forge.test.models.utils import build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_squeezebert_sequence_classification_pytorch(test_device):
    # Load Bart tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, "squeezebert/squeezebert-mnli")
    model = download_model(AutoModelForSequenceClassification.from_pretrained, "squeezebert/squeezebert-mnli")

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Example from multi-nli validation set
    text = """Hello, my dog is cute"""

    # Data preprocessing
    input_tokens = tokenizer.encode(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens]
    module_name = build_module_name(
        framework="pt", model="squeezebert", variant=variant, task="sequence_classification"
    )
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
