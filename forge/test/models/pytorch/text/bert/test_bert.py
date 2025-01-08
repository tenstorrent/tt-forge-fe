# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
)

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Task, build_module_name
from test.utils import download_model


def generate_model_bert_maskedlm_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    tokenizer = BertTokenizer.from_pretrained(variant)
    model = BertForMaskedLM.from_pretrained(variant)

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
@pytest.mark.parametrize("variant", ["bert-base-uncased"])
def test_bert_masked_lm_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="bert", variant=variant, task=Task.MASKED_LM)

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_bert_maskedlm_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_bert_qa_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    model = download_model(BertForQuestionAnswering.from_pretrained, variant)

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
@pytest.mark.parametrize("variant", ["bert-large-cased-whole-word-masking-finetuned-squad"])
def test_bert_question_answering_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="bert", variant=variant, task=Task.QA)

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_bert_qa_hf_pytorch()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_bert_seqcls_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    model = download_model(BertForSequenceClassification.from_pretrained, variant)

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
@pytest.mark.parametrize("variant", ["textattack/bert-base-uncased-SST-2"])
def test_bert_sequence_classification_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="bert", variant=variant, task=Task.SEQUENCE_CLASSIFICATION
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_bert_seqcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


def generate_model_bert_tkcls_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    model = download_model(BertForTokenClassification.from_pretrained, variant)

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
@pytest.mark.parametrize("variant", ["dbmdz/bert-large-cased-finetuned-conll03-english"])
def test_bert_token_classification_pytorch(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="bert", variant=variant, task=Task.TOKEN_CLASSIFICATION
    )

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_bert_tkcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
