# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.regnet.pytorch import ModelLoader, ModelVariant

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
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

variants = [
    ModelVariant.Y_040,
    ModelVariant.Y_064,
    ModelVariant.Y_080,
    ModelVariant.Y_120,
    ModelVariant.Y_160,
    ModelVariant.Y_320,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_regnet_img_classification(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.REGNET,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model.config.return_dict = False
    input_tensor = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_tensor]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification and Inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    loader.post_processing(co_out)


variants = [
    ModelVariant.Y_400MF,
    ModelVariant.Y_800MF,
    ModelVariant.Y_1_6GF,
    ModelVariant.Y_3_2GF,
    ModelVariant.Y_8GF,
    ModelVariant.Y_16GF,
    ModelVariant.Y_32GF,
    pytest.param(ModelVariant.Y_128GF, marks=[pytest.mark.out_of_memory, pytest.mark.xfail]),
    ModelVariant.X_400MF,
    ModelVariant.X_800MF,
    ModelVariant.X_1_6GF,
    ModelVariant.X_3_2GF,
    ModelVariant.X_8GF,
    ModelVariant.X_16GF,
    ModelVariant.X_32GF,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_regnet_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.REGNET,
        variant=variant,
        task=Task.CV_IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )
    if variant == ModelVariant.Y_128GF:
        pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-onnx/issues/2949")

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

    verify_cfg = VerifyConfig()
    if variant == ModelVariant.X_8GF:
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Run model on sample data and print results
    loader.post_processing(co_out)
