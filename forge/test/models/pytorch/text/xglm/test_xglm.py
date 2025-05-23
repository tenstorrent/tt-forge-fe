# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoTokenizer, XGLMConfig, XGLMForCausalLM

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.utils import download_model

variants = [
    "facebook/xglm-564M",
    "facebook/xglm-1.7B",
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_xglm_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model="xglm", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    config = XGLMConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = XGLMConfig(**config_dict)

    framework_model = download_model(XGLMForCausalLM.from_pretrained, variant, config=config)

    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = "My name is Thomas and my main"
    input_tokens = tokenizer(
        prefix_text,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
