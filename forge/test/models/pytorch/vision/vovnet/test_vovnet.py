# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.vovnet.model_utils.model_utils import (
    get_image,
    preprocess_steps,
    preprocess_timm_model,
)
from test.models.pytorch.vision.vovnet.model_utils.src_vovnet_stigma import (
    vovnet39,
    vovnet57,
)
from test.utils import download_model

varaints = [
    pytest.param("vovnet27s", marks=pytest.mark.push),
    "vovnet39",
    "vovnet57",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", varaints)
def test_vovnet_osmr_pytorch(variant):

    if variant in ["vovnet27s"]:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VOVNET,
        variant=variant,
        source=Source.OSMR,
        task=Task.IMAGE_CLASSIFICATION,
        group=group,
        priority=priority,
    )

    # Load model
    framework_model = download_model(ptcv_get_model, variant, pretrained=True).to(torch.bfloat16)

    # prepare input
    image_tensor = get_image()
    inputs = [image_tensor.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    verify_cfg = VerifyConfig()
    if variant == "vovnet39":
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98))

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=verify_cfg,
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


def generate_model_vovnet39_imgcls_stigma_pytorch():
    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet39)
    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
def test_vovnet_v1_39_stigma_pytorch():

    variant = "vovnet39"

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VOVNETV1,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

    framework_model, inputs, _ = generate_model_vovnet39_imgcls_stigma_pytorch()

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


def generate_model_vovnet57_imgcls_stigma_pytorch():
    # STEP 2: Create Forge module from PyTorch model
    model, image_tensor = download_model(preprocess_steps, vovnet57)

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


@pytest.mark.nightly
def test_vovnet_v1_57_stigma_pytorch():

    variant = "vovnet_v1_57"

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VOVNET,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
    )

    framework_model, inputs, _ = generate_model_vovnet57_imgcls_stigma_pytorch()

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


def generate_model_vovnet_imgcls_timm_pytorch(variant):
    model, image_tensor = download_model(preprocess_timm_model, variant)

    return model.to(torch.bfloat16), [image_tensor.to(torch.bfloat16)], {}


variants = [
    "ese_vovnet19b_dw",
    "ese_vovnet39b",
    "ese_vovnet99b",
    pytest.param(
        "ese_vovnet19b_dw.ra_in1k",
        marks=[pytest.mark.xfail],
    ),
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_vovnet_timm_pytorch(variant):

    if variant == "ese_vovnet19b_dw.ra_in1k":
        group = (ModelGroup.RED,)
        priority = (ModelPriority.P1,)
    else:
        group = (ModelGroup.GENERALITY,)
        priority = (ModelPriority.P2,)

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.VOVNET,
        variant=variant,
        source=Source.TORCH_HUB,
        task=Task.OBJECT_DETECTION,
        group=group,
        priority=priority,
    )

    framework_model, inputs, _ = generate_model_vovnet_imgcls_timm_pytorch(
        variant,
    )

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
