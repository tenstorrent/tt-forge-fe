# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import paddle
import pytest
from paddlenlp.transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from forge.forge_property_utils import Framework, Source, Task, build_module_name

variants = ["facebook/llama-7b"]


@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_llama_decoder(variant, forge_property_recorder):
    full_model = LlamaForCausalLM.from_pretrained(variant)
    decoder = full_model.llama.layers[0]
    hidden_size = full_model.llama.hidden_size
    inputs = [paddle.randn([1, 1, hidden_size])]

    class LlamaDecoderWrapper(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states):
            return self.model(hidden_states=hidden_states)

    model = LlamaDecoderWrapper(decoder)

    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_llama_rms_norm(variant, forge_property_recorder):
    full_model = LlamaModel.from_pretrained(variant)
    rms_norm = full_model.llama.norm
    hidden_size = full_model.llama.hidden_size
    inputs = [paddle.randn([1, 1, hidden_size])]

    class LlamaRMSNormWrapper(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states):
            return self.model(hidden_states=hidden_states)

    model = LlamaRMSNormWrapper(rms_norm)

    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)


@pytest.mark.parametrize("variant", variants)
def test_llama_lm_head(variant, forge_property_recorder):
    full_model = LlamaForCausalLM.from_pretrained(variant)
    lm_head = full_model.lm_head
    hidden_size = full_model.llama.hidden_size
    inputs = [paddle.randn([1, 1, hidden_size])]

    class LlamaLMHeadWrapper(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, hidden_states):
            return self.model(hidden_states=hidden_states)

    model = LlamaLMHeadWrapper(lm_head)

    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)
