# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    DistilBertForMaskedLM,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
)
from transformers.modeling_outputs import (
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model


# Wrapper to return tensor outputs
class DistilBertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Case 1: Question Answering
        if isinstance(output, QuestionAnsweringModelOutput):
            return output.start_logits, output.end_logits

        # Case 2: Token Classification, MaskedLM, Sequence Classification
        if isinstance(output, (TokenClassifierOutput, MaskedLMOutput, SequenceClassifierOutput)):
            return output.logits


variants = [
    pytest.param("distilbert-base-cased", marks=[pytest.mark.push]),
    pytest.param("distilbert-base-uncased"),
    pytest.param("distilbert-base-multilingual-cased"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_distilbert_masked_lm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DISTILBERT,
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Load DistilBert tokenizer and model from HuggingFace
    # Variants: distilbert-base-uncased, distilbert-base-cased,
    # distilbert-base-multilingual-cased
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, variant)
    framework_model = download_model(DistilBertForMaskedLM.from_pretrained, variant)
    framework_model = DistilBertWrapper(framework_model)

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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post-processing
    logits = co_out[0]
    mask_token_index = (input_tokens["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["distilbert-base-cased-distilled-squad"])
def test_distilbert_question_answering_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DISTILBERT,
        variant=variant,
        task=Task.QA,
        source=Source.HUGGINGFACE,
    )

    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, variant)
    framework_model = download_model(DistilBertForQuestionAnswering.from_pretrained, variant)
    framework_model = DistilBertWrapper(framework_model)

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

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    answer_start_index = co_out[0].argmax()
    answer_end_index = co_out[1].argmax()

    predict_answer_tokens = input_tokens.input_ids[0, answer_start_index : answer_end_index + 1]
    print("predicted answer ", tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["distilbert-base-uncased-finetuned-sst-2-english"])
def test_distilbert_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DISTILBERT,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load DistilBert tokenizer and model from HuggingFace
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, variant)
    framework_model = download_model(DistilBertForSequenceClassification.from_pretrained, variant)
    framework_model = DistilBertWrapper(framework_model)

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

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_class_id = co_out[0].argmax().item()
    predicted_category = framework_model.config.id2label[predicted_class_id]

    print(f"predicted category: {predicted_category}")


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["Davlan/distilbert-base-multilingual-cased-ner-hrl"])
def test_distilbert_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DISTILBERT,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load DistilBERT tokenizer and model from HuggingFace
    tokenizer = download_model(DistilBertTokenizer.from_pretrained, variant)
    framework_model = download_model(DistilBertForTokenClassification.from_pretrained, variant)
    framework_model = DistilBertWrapper(framework_model)

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
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_token_class_ids = co_out[0].argmax(-1)
    predicted_token_class_ids = torch.masked_select(predicted_token_class_ids, (input_tokens["attention_mask"][0] == 1))
    predicted_tokens_classes = [framework_model.config.id2label[t.item()] for t in predicted_token_class_ids]

    print(f"Context: {sample_text}")
    print(f"Answer: {predicted_tokens_classes}")
