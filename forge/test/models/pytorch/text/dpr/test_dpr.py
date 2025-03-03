# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRReader,
    DPRReaderTokenizer,
)

import forge
from forge.verify.config import VerifyConfig
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

params = [
    pytest.param("facebook/dpr-ctx_encoder-single-nq-base", marks=[pytest.mark.push]),
    pytest.param("facebook/dpr-ctx_encoder-multiset-base"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", params)
def test_dpr_context_encoder_pytorch(record_forge_property, variant):
    if variant != "facebook/dpr-ctx_encoder-single-nq-base":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="dpr",
        variant=variant,
        suffix="context_encoder",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-ctx_encoder-single-nq-base, facebook/dpr-ctx_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRContextEncoderTokenizer.from_pretrained, model_ckpt)
    framework_model = download_model(DPRContextEncoder.from_pretrained, model_ckpt, return_dict=False)

    # Load data sample
    sample_text = "Hello, is my dog cute?"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"], input_tokens["token_type_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(verify_values=False))

    # Inference
    embeddings = compiled_model(*inputs)

    # Results
    print("embeddings", embeddings)


variants = ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_question_encoder_pytorch(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="dpr",
        variant=variant,
        suffix="question_encoder",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-question_encoder-single-nq-base, facebook/dpr-question_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRQuestionEncoderTokenizer.from_pretrained, model_ckpt)
    framework_model = download_model(DPRQuestionEncoder.from_pretrained, model_ckpt, return_dict=False)

    # Load data sample
    sample_text = "Hello, is my dog cute?"

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"], input_tokens["token_type_ids"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify_values = True

    if variant == "facebook/dpr-question_encoder-multiset-base":
        verify_values = False

    # Model Verification
    verify(inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(verify_values=verify_values))

    # Inference
    embeddings = compiled_model(*inputs)

    # Results
    print("embeddings", embeddings)


variants = ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_reader_pytorch(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="dpr",
        variant=variant,
        suffix="reader",
        source=Source.HUGGINGFACE,
        task=Task.QA,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-reader-single-nq-base, facebook/dpr-reader-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRReaderTokenizer.from_pretrained, model_ckpt)
    framework_model = download_model(DPRReader.from_pretrained, model_ckpt, return_dict=False)

    # Data preprocessing
    input_tokens = tokenizer(
        questions=["What is love?"],
        titles=["Haddaway"],
        texts=["'What Is Love' is a song recorded by the artist Haddaway"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(verify_values=False))

    # Inference
    outputs = compiled_model(*inputs)

    # Post processing
    start_logits = outputs[0]
    end_logits = outputs[1]
    relevance_logits = outputs[2]

    start_indices = torch.argmax(start_logits, dim=1)
    end_indices = torch.argmax(end_logits, dim=1)

    answers = []
    for i, input_id in enumerate(inputs[0]):
        start_idx = start_indices[i].item()
        end_idx = end_indices[i].item()
        answer_tokens = input_id[start_idx : end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answers.append(answer)

    print("Predicted Answer:", answers[0])
