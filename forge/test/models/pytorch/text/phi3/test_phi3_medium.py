# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer, Phi3Config, Phi3ForCausalLM

import forge

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["microsoft/Phi-3-medium-128k-instruct"]


@pytest.mark.parametrize("variant", variants)
def test_phi3_causal_lm(variant):
    config = Phi3Config.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)
    tokenizer = AutoTokenizer.from_pretrained(variant)
    framework_model = Phi3ForCausalLM.from_pretrained(variant, config=config).to("cpu")
    framework_model.eval()
    input_prompt = "Africa is an emerging economy because"
    inputs = tokenizer(input_prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        op = framework_model(inputs["input_ids"], inputs["attention_mask"])
        logger.info(f"op={op}")
    module_name = build_module_name(variant, Source.HUGGINGFACE, Framework.PYTORCH, Task.CAUSAL_LM)
    compiled_model = forge.compile(framework_model, [inputs["input_ids"], inputs["attention_mask"]], module_name)
