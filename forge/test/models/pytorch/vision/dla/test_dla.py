# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.dla.utils.utils import load_dla_model, post_processing
from test.models.pytorch.vision.utils.utils import load_timm_model_and_input

variants = [
    "dla34",
    "dla46_c",
    "dla46x_c",
    "dla60",
    "dla60x",
    "dla60x_c",
    "dla102",
    "dla102x",
    "dla102x2",
    "dla169",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_dla_pytorch(forge_property_recorder, variant):
    if variant != "dla34":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="dla", variant=variant, task=Task.VISUAL_BACKBONE, source=Source.TORCHVISION
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and prepare input data
    framework_model, inputs = load_dla_model(variant)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Inference
    output = compiled_model(*inputs)

    # post processing
    post_processing(output)


variants = ["dla34.in1k"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_dla_timm(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="dla",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load the model and inputs
    framework_model, inputs = load_timm_model_and_input(variant)
    framework_model.to(torch.bfloat16)
    inputs.to(torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
