import paddle
import pytest
from paddlenlp.transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["facebook/llama-7b"]

@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_llama(variant):
    model = LlamaForCausalLM.from_pretrained(variant)
    tokenizer = LlamaTokenizer.from_pretrained(variant)
    text = "Once upon a time"
    inputs = tokenizer(text, return_tensors="pd")
    
    # Test framework model
    outputs = model(**inputs)
    logits = outputs[0]
    decoded_tokens = paddle.argmax(logits, axis=-1)
    decoded_text = tokenizer.decode(decoded_tokens[0].numpy().tolist(), skip_special_tokens=True)
    print(decoded_text)

    class LlamaWrapper(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, position_ids, attention_mask):
            return self.model(input_ids = input_ids, position_ids = position_ids, attention_mask = attention_mask)
        
    model = LlamaWrapper(model)

    inputs = [inputs["input_ids"], inputs["position_ids"], inputs["attention_mask"]]
    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)
    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)
    
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

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)
    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)

@pytest.mark.xfail() # float16
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

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)
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

    input_spec = [paddle.static.InputSpec(shape=inp.shape, dtype=inp.dtype) for inp in inputs]
    framework_model,_ = paddle_trace(model, input_spec)
    compiled_model = forge.compile(framework_model, inputs)
    verify(inputs, framework_model, compiled_model)