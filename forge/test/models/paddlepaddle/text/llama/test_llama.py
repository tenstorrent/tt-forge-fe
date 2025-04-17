# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import paddle
import pytest
from paddlenlp.transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from forge.forge_property_utils import Framework, Source, Task

variants = ["facebook/llama-7b"]


@pytest.mark.nightly
# @pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_llama(variant, forge_property_recorder):
    # Record Forge properties
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="llama",
        variant=variant[9:],
        source=Source.PADDLENLP,
        task=Task.CAUSAL_LM,
    )
    forge_property_recorder.record_group("generality")

    # Load Model and Tokenizer
    model = LlamaForCausalLM.from_pretrained(variant)
    tokenizer = LlamaTokenizer.from_pretrained(variant)

    # Load sample
    text = "Once upon a time"
    encoded_inputs = tokenizer(text, return_tensors="pd")
    inputs = [encoded_inputs["input_ids"], encoded_inputs["position_ids"], encoded_inputs["attention_mask"]]

    # Test framework model
    outputs = model(**encoded_inputs)
    logits = outputs[0]
    decoded_tokens = paddle.argmax(logits, axis=-1)
    decoded_text = tokenizer.decode(decoded_tokens[0].numpy().tolist(), skip_special_tokens=True)
    print(decoded_text)

    # Wrap Model to fix input signature
    class LlamaWrapper(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, position_ids, attention_mask):
            return self.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)

    model = LlamaWrapper(model)

    # Compile Model
    # framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(
        model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Verify
    verify(inputs, model, compiled_model, forge_property_handler=forge_property_recorder)
