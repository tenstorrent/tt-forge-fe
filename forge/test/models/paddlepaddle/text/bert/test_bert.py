# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from test.models.utils import Framework, Source, Task, build_module_name

from paddlenlp.transformers import BertForSequenceClassification, BertForMaskedLM, BertForQuestionAnswering, BertTokenizer

inputs_map = {
    "bert-base-uncased": {
        "sequence": ["Hello, my dog is cute"],
        "mask": ["One, [MASK], three, four"],
        "question": ["What is the capital of China?"]
    },
    "cl-tohoku/bert-base-japanese": {
        "sequence": ["こんにちは、私の犬はかわいいです"],
        "mask": ["一つ、[MASK]、三、四"],
        "question": ["中国の首都はどこですか？"]
    },
    "uer/chinese-roberta-base": {
        "sequence": ["你好，我的狗很可爱"],
        "mask": ["一，[MASK]，三，四"],
        "question": ["中国的首都是哪里？"]
    },
}

@pytest.mark.nightly
@pytest.mark.parametrize("variant, inputs_data", inputs_map.items())
def test_bert_sequence_classification(forge_property_recorder, variant, inputs_data):
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="bert",
        variant=variant.split("/")[-1] if "/" in variant else variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.PADDLENLP,
    )

    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    model = BertForSequenceClassification.from_pretrained(variant, num_classes=2)
    tokenizer = BertTokenizer.from_pretrained(variant)

    input = inputs_data["sequence"]
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)

    inputs = [
        paddle.to_tensor(value) for value in encoded_input.values()
    ]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)
    
    compiled_model = forge.compile(framework_model, inputs)

    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

@pytest.mark.nightly
@pytest.mark.parametrize("variant, inputs_data", inputs_map.items())
def test_bert_maskedlm(forge_property_recorder, variant, inputs_data):
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="bert",
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.PADDLENLP,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Model and Tokenizer
    model = BertForMaskedLM.from_pretrained(variant)
    tokenizer = BertTokenizer.from_pretrained(variant)

    # Load sample
    input = inputs_data["mask"]
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)

    inputs = [
        paddle.to_tensor(value) for value in encoded_input.values()
    ]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    outputs = compiled_model(*inputs)
    logits = outputs[0]
    mask_token_index = (inputs[0] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1).item()
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))

@pytest.mark.nightly
@pytest.mark.parametrize("variant, inputs_data", inputs_map.items())
def test_bert_question_answering(forge_property_recorder, variant, inputs_data):
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="bert",
        variant=variant,
        task=Task.QA,
        source=Source.PADDLENLP,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Model and Tokenizer
    model = BertForQuestionAnswering.from_pretrained(variant)
    tokenizer = BertTokenizer.from_pretrained(variant)

    # Load sample
    question = inputs
    encoded_input = tokenizer(question, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)

    inputs = [
        paddle.to_tensor(value) for value in encoded_input.values()
    ]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    outputs = compiled_model(*inputs)
    start_logits, end_logits = outputs
    start_index = start_logits.argmax(dim=-1).item()
    end_index = end_logits.argmax(dim=-1).item()
    answer = tokenizer.decode(encoded_input["input_ids"][0][start_index:end_index+1])
    print("The answer is: ", answer)
