# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle
from paddlenlp.transformers import (
    ErnieForSequenceClassification,
    ErnieForMaskedLM,
    ErnieForQuestionAnswering,
    ErnieTokenizer,
)

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

variants = ["ernie-1.0"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_ernie_for_sequence_classification(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.ERNIE,
        variant=variant[6:],
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.PADDLENLP,
    )

    # Load Model and Tokenizer
    model = ErnieForSequenceClassification.from_pretrained(variant, num_classes=2)
    tokenizer = ErnieTokenizer.from_pretrained(variant)

    # Load sample
    input = ["Hello, my dog is cute"]
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)
    inputs = [
        paddle.to_tensor(value) for value in encoded_input.values()
    ]  # [input_ids, token_type_ids, position_ids, attention_mask]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_ernie_maskedlm(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.ERNIE,
        variant=variant[6:],
        task=Task.MASKED_LM,
        source=Source.PADDLENLP,
    )

    # Load Model and Tokenizer
    model = ErnieForMaskedLM.from_pretrained(variant)
    tokenizer = ErnieTokenizer.from_pretrained(variant)

    # Load sample
    input = ["One, [MASK], three, four"]
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)
    inputs = [paddle.to_tensor(value) for value in encoded_input.values()]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)

    # Inference
    outputs = compiled_model(*inputs)
    logits = outputs[0]
    mask_token_index = (inputs[0] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1).item()
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_ernie_question_answering(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.ERNIE,
        variant=variant[6:],
        task=Task.QA,
        source=Source.PADDLENLP,
    )

    # Load Model and Tokenizer
    model = ErnieForQuestionAnswering.from_pretrained(variant)
    tokenizer = ErnieTokenizer.from_pretrained(variant)

    # Load sample
    question = ["What is the capital of China?"]
    encoded_input = tokenizer(
        question, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True
    )
    inputs = [paddle.to_tensor(value) for value in encoded_input.values()]

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)

    # Inference
    outputs = compiled_model(*inputs)
    start_logits, end_logits = outputs
    start_index = start_logits.argmax(dim=-1).item()
    end_index = end_logits.argmax(dim=-1).item()
    answer = tokenizer.decode(encoded_input["input_ids"][0][start_index : end_index + 1])
    print("The answer is: ", answer)
