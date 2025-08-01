# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from mlp_mixer_pytorch import MLPMixer
from third_party.tt_forge_models.mlp_mixer.pytorch import ModelLoader, ModelVariant

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

variants = [
    pytest.param(
        ModelVariant.MIXER_B16_224,
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2599")],
    ),
    pytest.param(
        ModelVariant.MIXER_B16_224_IN21K,
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2599")],
    ),
    pytest.param(ModelVariant.MIXER_B16_224_MIIL),
    pytest.param(
        ModelVariant.MIXER_B16_224_MIIL_IN21K,
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2599")],
    ),
    pytest.param(ModelVariant.MIXER_B32_224),
    pytest.param(
        ModelVariant.MIXER_L16_224,
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2599")],
    ),
    pytest.param(
        ModelVariant.MIXER_L16_224_IN21K,
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2599")],
    ),
    pytest.param(ModelVariant.MIXER_L32_224),
    pytest.param(ModelVariant.MIXER_S16_224),
    pytest.param(ModelVariant.MIXER_S32_224),
    pytest.param(
        ModelVariant.MIXER_B16_224_GOOG_IN21K,
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2599")],
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_mlp_mixer_timm_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MLPMIXER,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.print_cls_results(co_out)


@pytest.mark.nightly
def test_mlp_mixer_pytorch():

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.MLPMIXER,
        source=Source.GITHUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load model and input
    framework_model = MLPMixer(
        image_size=256,
        channels=3,
        patch_size=16,
        dim=512,
        depth=12,
        num_classes=1000,
    ).to(torch.bfloat16)
    framework_model.eval()

    inputs = [torch.randn(1, 3, 256, 256).to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
