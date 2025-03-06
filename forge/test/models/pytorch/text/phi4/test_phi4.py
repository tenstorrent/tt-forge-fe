# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoTokenizer, PhiConfig, PhiForCausalLM

import forge

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["microsoft/phi-4"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_phi_1_5_causal_lm(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(variant, Source.HUGGINGFACE, Framework.PYTORCH, Task.CAUSAL_LM)

    # Record Forge Property
    record_forge_property("group", "priority")

    # PhiConfig from pretrained variant, disable return_dict and caching.
    config = PhiConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = PhiConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(variant)
    framework_model = PhiForCausalLM.from_pretrained(variant, config=config).to("cpu")
    framework_model.eval()

    # input_prompt
    input_prompt = "Africa is an emerging economy because"

    # Tokenize input
    inputs = tokenizer(input_prompt, return_tensors="pt").to("cpu")

    sample_inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs, module_name)

    # Model Verification
    verify(sample_inputs, framework_model, compiled_model)
