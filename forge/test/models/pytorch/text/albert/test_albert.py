# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
from transformers import AlbertForMaskedLM, AlbertTokenizer, AlbertForTokenClassification
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name, Framework, Task
import torch
from forge.verify.verify import verify


sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("size", sizes, ids=sizes)
def test_albert_masked_lm_pytorch(record_forge_property, size, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="albert", variant=variant, task=Task.MASKED_LM)

    record_forge_property("module_name", module_name)

    model_ckpt = f"albert-{size}-{variant}"

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AlbertTokenizer.from_pretrained, model_ckpt)
    framework_model = download_model(AlbertForMaskedLM.from_pretrained, model_ckpt)

    # Load data sample
    sample_text = "The capital of France is [MASK]."

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)


sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.parametrize("size", sizes, ids=sizes)
def test_albert_token_classification_pytorch(record_forge_property, size, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="albert", variant=variant, task=Task.TOKEN_CLASSIFICATION
    )

    record_forge_property("module_name", module_name)

    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    # Variants: albert-base-v1, albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1
    # albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2
    model_ckpt = f"albert-{size}-{variant}"

    # Load ALBERT tokenizer and model from HuggingFace
    tokenizer = AlbertTokenizer.from_pretrained(model_ckpt)
    framework_model = AlbertForTokenClassification.from_pretrained(model_ckpt)

    # Load data sample
    sample_text = "HuggingFace is a company based in Paris and New York"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)
