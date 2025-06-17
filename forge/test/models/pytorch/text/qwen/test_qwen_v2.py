# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2ForTokenClassification,
)

from torch import nn

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

# Variants for testing
variants = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct-1M",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct-1M",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail
@pytest.mark.nightly
def test_qwen_clm(variant):
    if variant in [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct-1M",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct-1M",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ]:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWENV2,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=group,
        priority=priority,
    )

    if variant in [
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct-1M",
        "Qwen/Qwen2.5-14B-Instruct-1M",
    ]:
        pytest.xfail(reason="Requires multi-chip support")

    # Load model and tokenizer
    framework_model = AutoModelForCausalLM.from_pretrained(variant, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(variant)

    # Prepare input
    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize and generate
    model_inputs = tokenizer([text], return_tensors="pt")
    inputs = [model_inputs["input_ids"]]
        
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids):
            return self.model(input_ids).logits
    
    framework_model = Wrapper(framework_model)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["Qwen/Qwen2-7B"])
def test_qwen2_token_classification(variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 32 GB during compile time)")

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWENV2,
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and tokenizer
    framework_model = Qwen2ForTokenClassification.from_pretrained(variant)
    framework_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(variant)

    # Prepare input
    text = "HuggingFace is a company based in Paris and New York."
    model_inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")

    inputs = [model_inputs["input_ids"], model_inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/QVQ-72B-Preview",
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.xfail
def test_qwen2_conditional_generation(variant):

    # Record Forge Property
    record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.QWENV2,
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    if variant in ["Qwen/Qwen2.5-VL-72B-Instruct", "Qwen/QVQ-72B-Preview"]:
        raise RuntimeError("Requires transformers>=4.49.0 and multi-chip support")
    else:
        raise RuntimeError("Requires transformers>=4.49.0")
