# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge

from test.models.pytorch.multimodal.deepseek.utils.models import (
    generate_model_deepseek_vl_pytorch,
    generation,
)
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.parametrize("variant", ["deepseek-ai/deepseek-vl-1.3b-base"])
def test_deepseek_vl_no_cache_cpu_pytorch(record_forge_property, variant):

    framework_model, vl_gpt, tokenizer, inputs_embeds = generate_model_deepseek_vl_pytorch(variant)
    answer = generation(
        max_new_tokens=512, model=framework_model, inputs_embeds=inputs_embeds, tokenizer=tokenizer, vl_gpt=vl_gpt
    )

    print(f"{prepare_inputs['sft_format'][0]}", answer)


@pytest.mark.parametrize("variant", ["deepseek-ai/deepseek-vl-1.3b-base"])
def test_deepseek_vl_pytorch(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="deepseek", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    framework_model, vl_gpt, tokenizer, inputs_embeds = generate_model_deepseek_vl_pytorch(variant)
    compiled_model = forge.compile(framework_model, sample_inputs=[inputs_embeds], module_name=module_name)
    answer = generation(
        max_new_tokens=1, model=compiled_model, inputs_embeds=inputs_embeds, tokenizer=tokenizer, vl_gpt=vl_gpt
    )

    print(f"{prepare_inputs['sft_format'][0]}", answer)
