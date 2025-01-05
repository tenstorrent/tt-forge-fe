# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from test.utils import download_model
import forge
from transformers import AutoTokenizer, XGLMForCausalLM, XGLMConfig
from test.models.utils import build_module_name


variants = ["facebook/xglm-564M", "facebook/xglm-1.7B"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_xglm_causal_lm(variant):
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
    module_name = build_module_name(framework="pt", model="xglm", variant=variant, task="causal_lm")
    compiled_model = forge.compile(model, sample_inputs=inputs, module_name=module_name)
