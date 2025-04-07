# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from test.models.utils import Framework, Source, Task, build_module_name

from paddlenlp.transformers import GLMTokenizer, GLMForConditionalGeneration

variants = ["THUDM/glm-515m", "THUDM/glm-2b", "THUDM/glm-large-chinese"]


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_glm(variant, forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="glm",
        variant=variant[10:],
        source=Source.PADDLENLP,
        task=Task.CONDITIONAL_GENERATION,
    )
    forge_property_recorder.record_model_name(module_name)
    forge_property_recorder.record_group("generality")

    # Load Model and Tokenizer
    model = GLMForConditionalGeneration.from_pretrained(variant)
    tokenizer = GLMTokenizer.from_pretrained(variant)

    # Load sample
    text = ["写一首关于大海的诗"]
    encoded_inputs = tokenizer(text, return_tensors="pd")
    inputs = [encoded_inputs["input_ids"]]

    # Test framework model
    outputs = model(*inputs)
    logits = outputs[0]
    decoded_tokens = paddle.argmax(logits, axis=-1)
    decoded_text = tokenizer.decode(decoded_tokens[0].numpy().tolist(), skip_special_tokens=True)
    print(decoded_text)

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        framework_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Verify
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
