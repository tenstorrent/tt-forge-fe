# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from transformers import AutoTokenizer, XGLMConfig, XGLMForCausalLM

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Task, build_module_name
from test.utils import download_model

variants = ["facebook/xglm-564M", "facebook/xglm-1.7B"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xglm_causal_lm(record_forge_property, variant):
    if variant != "facebook/xglm-564M":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="xglm", variant=variant, task=Task.CAUSAL_LM)

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Skip erase inverse ops in forge passess
    os.environ["FORGE_DISABLE_ERASE_INVERSE_OPS_PASS"] = "1"

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
