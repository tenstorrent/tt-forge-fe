# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from test.utils import download_model
import pytest
import forge
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    BertForTokenClassification,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)
import torch
from forge.verify.compare import compare_with_golden


def generate_model_bert_maskedlm_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = variant
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    model = BertForMaskedLM.from_pretrained(model_ckpt)

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object

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

    return model, [input_tokens["input_ids"]], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="TT_FATAL(weights.get_dtype() == DataType::BFLOAT16) in embedding op")
def test_bert_masked_lm_pytorch(test_device):
    model, inputs, _ = generate_model_bert_maskedlm_hf_pytorch("bert-base-uncased")

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_bert_masked_lm")

    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


def generate_model_bert_qa_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = variant
    tokenizer = download_model(BertTokenizer.from_pretrained, model_ckpt)
    model = download_model(BertForQuestionAnswering.from_pretrained, model_ckpt)

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object

    # Load data sample from SQuADv1.1
    context = """Super Bowl 50 was an American football game to determine the champion of the National Football League
    (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the
    National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title.
    The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
    As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed
    initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals
    (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently
    feature the Arabic numerals 50."""

    question = "Which NFL team represented the AFC at Super Bowl 50?"

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=384,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return model, [input_tokens["input_ids"]], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="TT_FATAL(weights.get_dtype() == DataType::BFLOAT16) in embedding op")
def test_bert_question_answering_pytorch(test_device):
    model, inputs, _ = generate_model_bert_qa_hf_pytorch("bert-large-cased-whole-word-masking-finetuned-squad")

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_bert_qa")

    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


def generate_model_bert_seqcls_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = variant
    tokenizer = download_model(BertTokenizer.from_pretrained, model_ckpt)
    model = download_model(BertForSequenceClassification.from_pretrained, model_ckpt)

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load data sample
    review = "the movie was great!"

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return model, [input_tokens["input_ids"]], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_bert_sequence_classification_pytorch(test_device):
    model, inputs, _ = generate_model_bert_seqcls_hf_pytorch(
        "textattack/bert-base-uncased-SST-2",
    )

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_bert_sequence_classification")

    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


def generate_model_bert_tkcls_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = variant
    tokenizer = download_model(BertTokenizer.from_pretrained, model_ckpt)
    model = download_model(BertForTokenClassification.from_pretrained, model_ckpt)

    compiler_cfg = forge.config._get_global_compiler_config()  # load global compiler config object

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

    return model, [input_tokens["input_ids"]], {}


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="TT_FATAL(weights.get_dtype() == DataType::BFLOAT16) in embedding op")
def test_bert_token_classification_pytorch(test_device):
    model, inputs, _ = generate_model_bert_tkcls_hf_pytorch("dbmdz/bert-large-cased-finetuned-conll03-english")

    compiled_model = forge.compile(model, sample_inputs=inputs, module_name="pt_bert_sequence_classification")

    co_out = compiled_model(*inputs)
    fw_out = model(*inputs)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
