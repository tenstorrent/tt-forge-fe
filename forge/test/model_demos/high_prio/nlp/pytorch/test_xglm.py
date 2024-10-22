# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
from transformers import AutoTokenizer, XGLMForCausalLM, XGLMConfig


variants = ["facebook/xglm-564M", "facebook/xglm-1.7B"]


@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xglm_causal_lm(variant, test_device):
    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    config = XGLMConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = XGLMConfig(**config_dict)
    model = download_model(XGLMForCausalLM.from_pretrained, variant, config=config)
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
    compiled_model = forge.compile(model, sample_inputs=inputs)
