# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
from forge.test.models.utils import build_module_name
from transformers import (
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    DistilBertForQuestionAnswering,
    DistilBertForTokenClassification,
    DistilBertForSequenceClassification,
)

variants = ["distilbert-base-uncased", "distilbert-base-cased", "distilbert-base-multilingual-cased"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_distilbert_masked_lm_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="distilbert", variant=variant, task="mlm")

    record_forge_property("module_name", module_name)

    # Load DistilBert tokenizer and model from HuggingFace
    # Variants: distilbert-base-uncased, distilbert-base-cased,
    # distilbert-base-multilingual-cased
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, variant)
    model = download_model(DistilBertForMaskedLM.from_pretrained, variant)

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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_distilbert_question_answering_pytorch(record_forge_property):
    module_name = build_module_name(framework="pt", model="distilbert", variant=model_ckpt, task="qa")

    record_forge_property("module_name", module_name)

    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = "distilbert-base-cased-distilled-squad"
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, model_ckpt)
    model = download_model(DistilBertForQuestionAnswering.from_pretrained, model_ckpt)

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

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_distilbert_sequence_classification_pytorch(record_forge_property):
    module_name = build_module_name(framework="pt", model="distilbert", variant=model_ckpt, task="seqcls")

    record_forge_property("module_name", module_name)

    # Load DistilBert tokenizer and model from HuggingFace
    model_ckpt = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, model_ckpt)
    model = download_model(DistilBertForSequenceClassification.from_pretrained, model_ckpt)

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

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)


@pytest.mark.nightly
@pytest.mark.model_analysis
def test_distilbert_token_classification_pytorch(record_forge_property):
    module_name = build_module_name(framework="pt", model="distilbert", variant=model_ckpt, task="token_cls")

    record_forge_property("module_name", module_name)
    # Load DistilBERT tokenizer and model from HuggingFace
    model_ckpt = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, model_ckpt)
    model = download_model(DistilBertForTokenClassification.from_pretrained, model_ckpt)

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
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
