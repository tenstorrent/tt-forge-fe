# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)

variants = ["meta-llama/Llama-3.2-11B-Vision-Instruct", "meta-llama/Llama-3.2-90B-Vision-Instruct"]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
@pytest.mark.xfail
def test_llama_vision_Instruct(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.LLAMA3_2,
        variant=variant,
        task=Task.MULTIMODAL_TEXT_GENERATION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    raise RuntimeError("Requires multi-chip support")
