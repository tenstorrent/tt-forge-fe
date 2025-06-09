# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import (
    BertForMaskedLM,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    BertTokenizer,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.text.bert.model_utils.utils import mean_pooling
from test.utils import download_model


class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        token_type_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        return self.model(input_ids, attention_mask, token_type_ids, position_ids)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["bert-base-uncased"])
@pytest.mark.push
def test_bert_masked_lm_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.HUGGINGFACE,
    )

    # Load Bert tokenizer and model from HuggingFace
    tokenizer = BertTokenizer.from_pretrained(variant)
    framework_model = BertForMaskedLM.from_pretrained(variant, return_dict=False)
    framework_model = BertWrapper(framework_model)

    # Load data sample
    sample_text = "The capital of France is [MASK]."

    # Data preprocessing
    tokenized = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [tokenized["input_ids"], tokenized["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # post processing
    logits = co_out[0]
    mask_token_index = (tokenized["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))


def generate_model_bert_qa_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    model = download_model(BertForQuestionAnswering.from_pretrained, variant, return_dict=False)
    framework_model = BertWrapper(model)

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

    return framework_model, inputs, tokenizer


variants = [
    pytest.param("phiyodr/bert-large-finetuned-squad2", marks=[pytest.mark.push]),
    pytest.param("bert-large-cased-whole-word-masking-finetuned-squad"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_bert_question_answering_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.BERT, variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    framework_model, inputs, tokenizer = generate_model_bert_qa_hf_pytorch(variant)
    framework_model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # post processing
    start_logits = co_out[0]
    end_logits = co_out[1]

    answer_start_index = start_logits.argmax()
    answer_end_index = end_logits.argmax()

    input_ids = inputs[0]
    predict_answer_tokens = input_ids[0, answer_start_index : answer_end_index + 1]

    print("predicted answer ", tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))


def generate_model_bert_seqcls_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    model = download_model(BertForSequenceClassification.from_pretrained, variant, return_dict=False)
    framework_model = BertWrapper(model)

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

    return framework_model, [input_tokens["input_ids"]], {}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["textattack/bert-base-uncased-SST-2"])
def test_bert_sequence_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    framework_model, inputs, _ = generate_model_bert_seqcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    predicted_value = co_out[0].argmax(-1).item()

    # Answer - "positive"
    print(f"Predicted Sentiment: {framework_model.config.id2label[predicted_value]}")


def generate_model_bert_tkcls_hf_pytorch(variant):
    # Load Bert tokenizer and model from HuggingFace
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    model = download_model(BertForTokenClassification.from_pretrained, variant, return_dict=False)

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

    return model, sample_text, [input_tokens["input_ids"]], input_tokens


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["dbmdz/bert-large-cased-finetuned-conll03-english"])
def test_bert_token_classification_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    framework_model, sample_text, inputs, input_tokens = generate_model_bert_tkcls_hf_pytorch(variant)
    framework_model = BertWrapper(framework_model)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # post processing
    predicted_token_class_ids = co_out[0].argmax(-1)
    predicted_token_class_ids = torch.masked_select(predicted_token_class_ids, (input_tokens["attention_mask"][0] == 1))
    predicted_tokens_classes = [framework_model.config.id2label[t.item()] for t in predicted_token_class_ids]

    print(f"Context: {sample_text}")
    print(f"Answer: {predicted_tokens_classes}")


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize("variant", ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"])
def test_bert_sentence_embedding_generation_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.BERT,
        variant=variant,
        task=Task.SENTENCE_EMBEDDING_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and tokenizer
    tokenizer = download_model(BertTokenizer.from_pretrained, variant)
    framework_model = download_model(BertModel.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()

    # prepare input
    sentence = "Bu örnek bir cümle"
    encoding = tokenizer(sentence, padding="max_length", truncation=True, max_length=16, return_tensors="pt")

    # Manually construct token_type_ids and position_ids
    batch_size, seq_len = encoding["input_ids"].shape
    encoding["token_type_ids"] = torch.zeros((batch_size, seq_len), dtype=torch.long)
    encoding["position_ids"] = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Inputs for forward pass
    inputs = [
        encoding["input_ids"],
        encoding["attention_mask"],
        encoding["token_type_ids"],
        encoding["position_ids"],
    ]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    sentence_embeddings = mean_pooling(co_out, encoding["attention_mask"])

    print("Sentence embeddings:", sentence_embeddings)
