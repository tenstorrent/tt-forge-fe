# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model
from third_party.tt_forge_models.hrnet.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.tools.utils import get_file
from torchvision import transforms

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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from test.models.models_utils import print_cls_results
from test.utils import download_model


def generate_model_hrnet_imgcls_osmr_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    """
    models = [
        hrnet_w18_small_v1,
        hrnet_w18_small_v2,
        hrnetv2_w18,
        hrnetv2_w30,
        hrnetv2_w32,
        hrnetv2_w40,
        hrnetv2_w44,
        hrnetv2_w48,
        hrnetv2_w64,
    ]
    """
    model = download_model(ptcv_get_model, variant, pretrained=True)
    model.eval()

    # Model load
    try:
        file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
        input_image = Image.open(file_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        input_batch = torch.rand(1, 3, 224, 224)
    print(input_batch.shape)

    return model.to(torch.bfloat16), [input_batch.to(torch.bfloat16)], {}


variants = [
    pytest.param("hrnet_w18_small_v1", marks=pytest.mark.push),
    pytest.param("hrnet_w18_small_v2"),
    pytest.param("hrnetv2_w18"),
    pytest.param("hrnetv2_w30"),
    pytest.param(
        "hrnetv2_w32",
    ),
    pytest.param(
        "hrnetv2_w40",
    ),
    pytest.param("hrnetv2_w44"),
    pytest.param(
        "hrnetv2_w48",
    ),
    pytest.param("hrnetv2_w64"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_hrnet_osmr_pytorch(variant):

    pcc = 0.99
    if variant in ["hrnetv2_w44", "hrnetv2_w64"]:
        pcc = 0.97

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.HRNET,
        variant=variant,
        source=Source.OSMR,
        task=Task.IMAGE_CLASSIFICATION,
    )

    framework_model, inputs, _ = generate_model_hrnet_imgcls_osmr_pytorch(
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
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0])


variants = [
    ModelVariant.HRNET_W18_SMALL,
    ModelVariant.HRNET_W18_SMALL_V2,
    ModelVariant.HRNET_W18,
    ModelVariant.HRNET_W30,
    ModelVariant.HRNET_W32,
    ModelVariant.HRNET_W40,
    ModelVariant.HRNET_W44,
    ModelVariant.HRNET_W48,
    ModelVariant.HRNET_W64,
    ModelVariant.HRNET_W18_MS_AUG_IN1K,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_hrnet_timm_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.HRNET,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    if variant in [
        ModelVariant.HRNET_W32,
        ModelVariant.HRNET_W40,
        ModelVariant.HRNET_W44,
        ModelVariant.HRNET_W48,
        ModelVariant.HRNET_W64,
        ModelVariant.HRNET_W18_MS_AUG_IN1K,
    ]:
        pytest.xfail(reason="Requires multi-chip support")

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

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Run model on sample data and print results
    loader.print_cls_results(co_out)
