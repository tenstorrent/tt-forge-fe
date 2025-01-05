# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig, OPTForQuestionAnswering, OPTForSequenceClassification
from test.models.utils import build_module_name, Framework, Task, Source


variants = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_opt_causal_lm(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="opt", variant=variant, task=Task.CAUSAL_LM)

    record_forge_property("module_name", module_name)

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"

    config = OPTConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = OPTConfig(**config_dict)
    model = download_model(OPTForCausalLM.from_pretrained, variant, config=config)
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
    compiled_model = forge.compile(
        model,
        inputs,
        module_name,
    )


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_opt_qa(record_forge_property, variant):
    module_name = build_module_name(framework=Framework.PYTORCH, model="opt", variant=variant, task=Task.QA)

    record_forge_property("module_name", module_name)

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(OPTForQuestionAnswering.from_pretrained, variant, torchscript=True)

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
    compiled_model = forge.compile(
        model,
        inputs,
        module_name,
    )


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_opt_sequence_classification(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="opt", variant=variant, task=Task.SEQUENCE_CLASSIFICATION
    )

    record_forge_property("module_name", module_name)

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    model = download_model(OPTForSequenceClassification.from_pretrained, variant, torchscript=True)

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
    compiled_model = forge.compile(
        model,
        inputs,
        module_name,
    )
