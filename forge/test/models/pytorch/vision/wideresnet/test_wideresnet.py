# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import urllib

import pytest
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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

from test.models.pytorch.vision.wideresnet.model_utils.utils import (
    generate_model_wideresnet_imgcls_pytorch,
    post_processing,
)
from test.utils import download_model

variants = [
    pytest.param("wide_resnet50_2", marks=[pytest.mark.push]),
    pytest.param("wide_resnet101_2"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_wideresnet_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.WIDERESNET,
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and prepare input data
    (framework_model, inputs) = generate_model_wideresnet_imgcls_pytorch(variant)
    framework_model.to(torch.bfloat16)
    inputs = [inputs[0].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    post_processing(co_out)


def generate_model_wideresnet_imgcls_timm(variant):
    # STEP 2: Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # STEP 3: Prepare input
    config = resolve_data_config({}, model=framework_model)
    transform = create_transform(**config)

    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    return framework_model.to(torch.bfloat16), [img_tensor.to(torch.bfloat16)]


variants = ["wide_resnet50_2", "wide_resnet101_2"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_wideresnet_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.WIDERESNET,
        source=Source.TIMM,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
    )

    (framework_model, inputs) = generate_model_wideresnet_imgcls_timm(variant)

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
    if variant == "wide_resnet50_2":
        verify_cfg = VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95))

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model, verify_cfg=verify_cfg)

    # Post processing
    post_processing(co_out)
