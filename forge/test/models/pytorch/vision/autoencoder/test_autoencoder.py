# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.autoencoder.pytorch import ModelLoader, ModelVariant

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


@pytest.mark.nightly
@pytest.mark.xfail
def test_conv_ae_pytorch():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.AUTOENCODER,
        variant=ModelVariant.CONV,
        task=Task.IMAGE_ENCODING,
        source=Source.GITHUB,
    )

    # Use loader for conv variant
    loader = ModelLoader(variant=ModelVariant.CONV)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    compiler_cfg = CompilerConfig(default_df_override=DataFormat.Float16_b)

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify(inputs, framework_model, compiled_model)


@pytest.mark.push
@pytest.mark.nightly
def test_linear_ae_pytorch():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.AUTOENCODER,
        variant=ModelVariant.LINEAR,
        task=Task.IMAGE_ENCODING,
        source=Source.GITHUB,
    )

    # Use loader for linear variant
    loader = ModelLoader(variant=ModelVariant.LINEAR)
    framework_model = loader.load_model()
    input_tensor = loader.load_inputs()
    inputs = [input_tensor]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing via loader
    save_path = "forge/test/models/pytorch/vision/autoencoder/results"
    loader.post_processing(co_out, save_path)
