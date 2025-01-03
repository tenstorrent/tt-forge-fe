# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRReader,
    DPRReaderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)
import torch
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name


variants = ["facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-ctx_encoder-multiset-base"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="TT_FATAL(weights.get_dtype() == DataType::BFLOAT16) in embedding op")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_context_encoder_pytorch(variant, record_property):
    module_name = build_module_name(framework="pt", model="dpr", variant=variant, task="context_encoder")

    record_property("frontend", "tt-forge-fe")
    record_property("module_name", module_name)

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-ctx_encoder-single-nq-base, facebook/dpr-ctx_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRContextEncoderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRContextEncoder.from_pretrained, model_ckpt)

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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co) for fo, co in zip(fw_out, co_out)])


variants = ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="TT_FATAL(weights.get_dtype() == DataType::BFLOAT16) in embedding op")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_question_encoder_pytorch(variant, record_property):
    module_name = build_module_name(framework="pt", model="dpr", variant=variant, task="question_encoder")

    record_property("frontend", "tt-forge-fe")
    record_property("module_name", module_name)

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-question_encoder-single-nq-base, facebook/dpr-question_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRQuestionEncoderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRQuestionEncoder.from_pretrained, model_ckpt)

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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co) for fo, co in zip(fw_out, co_out)])


variants = ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="TT_FATAL(weights.get_dtype() == DataType::BFLOAT16) in embedding op")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_reader_pytorch(variant, record_property):
    module_name = build_module_name(framework="pt", model="dpr", variant=variant, task="reader")

    record_property("frontend", "tt-forge-fe")
    record_property("module_name", module_name)

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-reader-single-nq-base, facebook/dpr-reader-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRReaderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRReader.from_pretrained, model_ckpt)

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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co) for fo, co in zip(fw_out, co_out)])
