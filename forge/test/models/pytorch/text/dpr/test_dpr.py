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
from forge.verify.verify import verify


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "facebook/dpr-ctx_encoder-single-nq-base",
            marks=pytest.mark.xfail(reason="Tensor mismatch. PCC = 0.966081010779316"),
        ),
        pytest.param(
            "facebook/dpr-ctx_encoder-multiset-base",
            marks=pytest.mark.xfail(reason="Tensor mismatch. PCC = 0.9652820314415684"),
        ),
    ],
)
def test_dpr_context_encoder_pytorch(variant, test_device):

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-ctx_encoder-single-nq-base, facebook/dpr-ctx_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRContextEncoderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRContextEncoder.from_pretrained, model_ckpt)
    model.config.return_dict = False

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object

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
    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )

    verify(inputs, model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("facebook/dpr-question_encoder-single-nq-base"),
        pytest.param(
            "facebook/dpr-question_encoder-multiset-base",
            marks=pytest.mark.xfail(reason="Tensor mismatch. PCC = 0.9871025782806984"),
        ),
    ],
)
def test_dpr_question_encoder_pytorch(variant, test_device):
    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-question_encoder-single-nq-base, facebook/dpr-question_encoder-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRQuestionEncoderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRQuestionEncoder.from_pretrained, model_ckpt)
    model.config.return_dict = False

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object

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
    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )

    verify(inputs, model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "facebook/dpr-reader-single-nq-base",
            marks=pytest.mark.xfail(reason="Tensor mismatch. PCC = 0.11557554999858398"),
        ),
        pytest.param(
            "facebook/dpr-reader-multiset-base",
            marks=pytest.mark.xfail(reason="Tensor mismatch. PCC = 0.199043936670584"),
        ),
    ],
)
def test_dpr_reader_pytorch(variant, test_device):
    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-reader-single-nq-base, facebook/dpr-reader-multiset-base
    model_ckpt = variant
    tokenizer = download_model(DPRReaderTokenizer.from_pretrained, model_ckpt)
    model = download_model(DPRReader.from_pretrained, model_ckpt)
    model.config.return_dict = False

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object

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
    compiled_model = forge.compile(
        model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )

    verify(inputs, model, compiled_model)
