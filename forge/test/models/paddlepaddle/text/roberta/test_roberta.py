# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import paddle

import forge
from forge.verify.verify import verify
from forge.tvm_calls.forge_utils import paddle_trace

from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties

from paddlenlp.transformers import (
    RobertaForSequenceClassification,
    RobertaForCausalLM,
    RobertaChineseTokenizer,
)

variants = ["hfl/rbt4"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_roberta_sequence_classification(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.ROBERTA,
        variant="rbt4-ch",
        task=Task.NLP_TEXT_CLS,
        source=Source.PADDLENLP,
    )

    # Load Model and Tokenizer
    tokenizer = RobertaChineseTokenizer.from_pretrained(variant)
    model = RobertaForSequenceClassification.from_pretrained(variant)

    # Load sample
    input = ["多么美好的一天"]
    encoded_input = tokenizer(input, return_tensors="pd")
    inputs = [encoded_input["input_ids"]]

    # Test framework model
    outputs = model(*inputs)
    print(outputs)

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail()
@pytest.mark.parametrize("variant", variants)
def test_roberta_causal_lm(variant):
    # Record Forge properties
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model=ModelArch.ROBERTA,
        variant="rbt4-ch",
        task=Task.NLP_FILL_MASK,
        source=Source.PADDLENLP,
    )

    # Load Model and Tokenizer
    tokenizer = RobertaChineseTokenizer.from_pretrained(variant)
    model = RobertaForCausalLM.from_pretrained(variant, ignore_mismatched_sizes=True)

    # Load sample
    input = ["这是一首关于海的诗"]
    inputs = tokenizer(input, return_tensors="pd")
    inputs = [inputs["input_ids"]]

    # Test framework model
    outputs = model(*inputs)
    logits = outputs[0]
    decoded_tokens = paddle.argmax(logits, axis=-1)
    decoded_text = tokenizer.decode(decoded_tokens.numpy().tolist(), skip_special_tokens=True)
    print(decoded_text)

    # Compile Model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs, module_name=module_name)

    # Verify
    verify(inputs, framework_model, compiled_model)
