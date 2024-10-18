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

variants = ["facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-ctx_encoder-multiset-base"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_context_encoder_pytorch(variant, test_device):

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-ctx_encoder-single-nq-base, facebook/dpr-ctx_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRContextEncoderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRContextEncoder.from_pretrained, model_ckpt)

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
    compiled_model = forge.compile(model, sample_inputs=inputs)


variants = ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_question_encoder_pytorch(variant, test_device):
    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-question_encoder-single-nq-base, facebook/dpr-question_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRQuestionEncoderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRQuestionEncoder.from_pretrained, model_ckpt)

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
    compiled_model = forge.compile(model, sample_inputs=inputs)


variants = ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_dpr_reader_pytorch(variant, test_device):
    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-reader-single-nq-base, facebook/dpr-reader-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRReaderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRReader.from_pretrained, model_ckpt)

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

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
    compiled_model = forge.compile(model, sample_inputs=inputs)
