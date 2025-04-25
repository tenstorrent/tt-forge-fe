# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    AlbertForMaskedLM,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertTokenizer,
    AutoTokenizer,
)

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.config import AutomaticValueChecker, VerifyConfig
from forge.verify.verify import verify

from test.utils import download_model

sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]


params = [
    pytest.param("base", "v1", marks=[pytest.mark.push]),
    pytest.param("large", "v1"),
    pytest.param("xlarge", "v1"),
    pytest.param("xxlarge", "v1"),
    pytest.param("base", "v2", marks=[pytest.mark.push]),
    pytest.param("large", "v2"),
    pytest.param("xlarge", "v2"),
    pytest.param("xxlarge", "v2"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size,variant", params)
def test_albert_masked_lm_pytorch(forge_property_recorder, size, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="albert",
        variant=f"{size}_{variant}",
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    model_ckpt = f"albert-{size}-{variant}"

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AlbertTokenizer.from_pretrained, model_ckpt)
    framework_model = download_model(AlbertForMaskedLM.from_pretrained, model_ckpt, return_dict=False)

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
    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        forge_property_handler=forge_property_recorder,
    )

    # Inference
    output = compiled_model(*inputs)

    # post processing
    logits = output[0]
    mask_token_index = (input_tokens.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))


# Task-specific models like AlbertForTokenClassification are pre-trained on general datasets.
# To make them suitable for specific tasks, they need to be fine-tuned on a labeled dataset for that task.

sizes = ["base", "large", "xlarge", "xxlarge"]
variants = ["v1", "v2"]


params = [
    pytest.param("base", "v1", marks=[pytest.mark.push]),
    pytest.param("large", "v1"),
    pytest.param("xlarge", "v1"),
    pytest.param("xxlarge", "v1"),
    pytest.param("base", "v2", marks=[pytest.mark.push]),
    pytest.param("large", "v2"),
    pytest.param("xlarge", "v2"),
    pytest.param("xxlarge", "v2"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("size,variant", params)
def test_albert_token_classification_pytorch(forge_property_recorder, size, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="albert",
        variant=f"{size}_{variant}",
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    # Variants: albert-base-v1, albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1
    # albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2
    model_ckpt = f"albert-{size}-{variant}"

    # Load ALBERT tokenizer and model from HuggingFace
    tokenizer = AlbertTokenizer.from_pretrained(model_ckpt)
    framework_model = AlbertForTokenClassification.from_pretrained(model_ckpt, return_dict=False)

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

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    if size == "xxlarge" and variant == "v2":
        pcc = 0.87
    elif size == "xlarge" and variant == "v2":
        pcc = 0.3
    else:
        pcc = 0.95

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )

    # Inference
    co_out = compiled_model(*inputs)

    # post processing
    predicted_token_class_ids = co_out[0].argmax(-1)
    predicted_token_class_ids = torch.masked_select(predicted_token_class_ids, (input_tokens["attention_mask"][0] == 1))
    predicted_tokens_classes = [framework_model.config.id2label[t.item()] for t in predicted_token_class_ids]

    print(f"Context: {sample_text}")
    print(f"Answer: {predicted_tokens_classes}")


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["twmkn9/albert-base-v2-squad2"])
def test_albert_question_answering_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="albert",
        variant=variant,
        task=Task.QA,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(AlbertForQuestionAnswering.from_pretrained, variant, return_dict=False)
    framework_model.eval()

    # Load data sample
    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    # Data preprocessing
    input_tokens = tokenizer(question, text, return_tensors="pt")
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.models
@pytest.mark.parametrize("variant", ["textattack/albert-base-v2-imdb"])
def test_albert_sequence_classification_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="albert",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load Albert tokenizer and model from HuggingFace
    tokenizer = download_model(AlbertTokenizer.from_pretrained, variant)
    framework_model = download_model(AlbertForSequenceClassification.from_pretrained, variant, return_dict=False)
    framework_model.eval()

    # Load data sample
    input_text = "Hello, my dog is cute."

    # Data preprocessing
    input_tokens = tokenizer(input_text, return_tensors="pt")
    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    co_out = compiled_model(*inputs)

    # post processing
    predicted_class_id = co_out[0].argmax().item()
    predicted_category = framework_model.config.id2label[predicted_class_id]

    print(f"predicted category: {predicted_category}")
