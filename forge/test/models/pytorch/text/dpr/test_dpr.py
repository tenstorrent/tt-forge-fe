# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
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

from test.models.utils import Framework, build_module_name
from test.utils import download_model

variants = ["facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-ctx_encoder-multiset-base"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_context_encoder_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="dpr", variant=variant, suffix="context_encoder")

    # Record Forge Property
    record_forge_property("module_name", module_name)

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


variants = ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_question_encoder_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="dpr", variant=variant, suffix="question_encoder"
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

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


variants = ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_reader_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="dpr", variant=variant, suffix="reader")

    # Record Forge Property
    record_forge_property("module_name", module_name)

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
