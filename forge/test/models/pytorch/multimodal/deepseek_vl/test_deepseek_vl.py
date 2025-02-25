# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge

from test.models.pytorch.multimodal.deepseek_vl.utils.load_model import (
    generate,
    generate_model_deepseek_vl_pytorch,
    verify_deepseek_vl,
)
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.parametrize("variant", ["deepseek-ai/deepseek-vl-1.3b-base"])
def test_deepseek_vl_no_cache_cpu_pytorch(record_forge_property, variant):

    framework_model, vl_gpt, tokenizer, inputs_embeds = generate_model_deepseek_vl_pytorch(variant)
    answer = generate(
        max_new_tokens=512, model=framework_model, inputs_embeds=inputs_embeds, tokenizer=tokenizer, vl_gpt=vl_gpt
    )
    print(answer)


@pytest.mark.parametrize("variant", ["deepseek-ai/deepseek-vl-1.3b-base"])
def test_deepseek_vl_pytorch(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="deepseek", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    framework_model, vl_gpt, tokenizer, inputs_embeds = generate_model_deepseek_vl_pytorch(variant)
    padded_inputs_embeds = torch.randn(1, 1140, 2048, dtype=torch.float32)
    compiled_model = forge.compile(framework_model, sample_inputs=[padded_inputs_embeds], module_name=module_name)
    verify_deepseek_vl(inputs_embeds, framework_model, compiled_model)
    answer = generate(
        max_new_tokens=512, model=compiled_model, inputs_embeds=inputs_embeds, tokenizer=tokenizer, vl_gpt=vl_gpt
    )

    print(answer)
