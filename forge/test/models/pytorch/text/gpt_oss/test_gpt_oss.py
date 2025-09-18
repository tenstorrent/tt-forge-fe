# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.text.gpt_oss.model_utils.model_utils import GPT_OSS_Wrapper


@pytest.mark.parametrize("variant", ["openai/gpt-oss-20b"])
def test_gpt_oss(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.GPT_OSS,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load Model and Tokenizer
    config = AutoConfig.from_pretrained(variant)

    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    model = AutoModelForCausalLM.from_pretrained(
        variant,
        config=config,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    framework_model = GPT_OSS_Wrapper(model)
    framework_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(variant)

    messages = [
        {"role": "user", "content": "How many rs are in the word 'strawberry'?"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cpu")

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[inputs],
        module_name=module_name,
    )

    # Model Verification
    verify([inputs], framework_model, compiled_model)
