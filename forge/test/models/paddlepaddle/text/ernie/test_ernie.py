# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from forge.tvm_calls.forge_utils import paddle_trace
import pytest
import paddle

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

from paddlenlp.transformers import (
    ErnieForSequenceClassification,
    ErnieForMaskedLM,
    ErnieTokenizer,
    AutoTokenizer,
    AutoModel,
)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["ernie-1.0"])
def test_ernie_for_sequence_classification(forge_property_recorder, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="ernie",
        variant=variant,
        task=Task.SEQUENCE_CLASSIFICATION,
        source=Source.PADDLENLP,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Model and Tokenizer
    model = ErnieForSequenceClassification.from_pretrained(variant, num_classes=2)
    tokenizer = ErnieTokenizer.from_pretrained(variant)

    # Load sample
    input = ["Hello, my dog is cute"]
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)

    inputs = [
        paddle.to_tensor(value) for value in encoded_input.values()
    ]  # [input_ids, token_type_ids, position_ids, attention_mask]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model, _ = paddle_trace(model, input_spec)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["ernie-1.0"])
def test_ernie_maskedlm(forge_property_recorder, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="ernie",
        variant=variant,
        task=Task.MASKED_LM,
        source=Source.PADDLENLP,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load Model and Tokenizer
    model = ErnieForMaskedLM.from_pretrained(variant)
    tokenizer = ErnieTokenizer.from_pretrained(variant)

    # Load sample
    input = ["One, [MASK], three, four"]
    encoded_input = tokenizer(input, return_token_type_ids=True, return_position_ids=True, return_attention_mask=True)

    inputs = [paddle.to_tensor(value) for value in encoded_input.values()]

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model, _ = paddle_trace(model, input_spec)

    # Compile Model
    compiled_model = forge.compile(framework_model, inputs)

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    outputs = compiled_model(*inputs)
    logits = outputs[0]
    mask_token_index = (inputs[0] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1).item()
    print("The predicted token for the [MASK] is: ", tokenizer.decode(predicted_token_id))
