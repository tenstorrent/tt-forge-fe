# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import timm
import torch
from loguru import logger
from PIL import Image
from third_party.tt_forge_models.efficientnet.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.tools.utils import get_file
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.utils import download_model

## https://huggingface.co/docs/timm/models/efficientnet

variants = [
    pytest.param(
        "efficientnet_b0",
        id="efficientnet_b0",
        marks=[pytest.mark.push],
    ),
    pytest.param(
        "efficientnet_b4",
        id="efficientnet_b4",
    ),
    pytest.param("hf_hub:timm/efficientnet_b0.ra_in1k", id="hf_hub_timm_efficientnet_b0_ra_in1k"),
    pytest.param("hf_hub:timm/efficientnet_b4.ra2_in1k", id="hf_hub_timm_efficientnet_b4_ra2_in1k"),
    pytest.param("hf_hub:timm/efficientnet_b5.in12k_ft_in1k", id="hf_hub_timm_efficientnet_b5_in12k_ft_in1k"),
    pytest.param("hf_hub:timm/tf_efficientnet_b0.aa_in1k", id="hf_hub_timm_tf_efficientnet_b0_aa_in1k"),
    pytest.param("hf_hub:timm/efficientnetv2_rw_s.ra2_in1k", id="hf_hub_timm_efficientnetv2_rw_s_ra2_in1k"),
    pytest.param("hf_hub:timm/tf_efficientnetv2_s.in21k", id="hf_hub_timm_tf_efficientnetv2_s_in21k"),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_timm(variant):
    if variant in ["efficientnet_b0"]:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.EFFICIENTNET,
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
        group=group,
        priority=priority,
    )

    # Load model
    framework_model = download_model(timm.create_model, variant, pretrained=True).to(torch.bfloat16)
    framework_model.eval()

    # Load and pre-process image
    try:
        if variant == "hf_hub:timm/tf_efficientnetv2_s.in21k":
            file_path = get_file(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            )
            img = Image.open(file_path).convert("RGB")
            use_1k_labels = False
        else:
            file_path = get_file("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
            img = Image.open(file_path).convert("RGB")
            use_1k_labels = True
        config = resolve_data_config({}, model=framework_model)
        transform = create_transform(**config)
        img_tensor = transform(img).unsqueeze(0)
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        img_tensor = torch.rand(1, 3, 224, 224)

    inputs = [img_tensor.to(torch.bfloat16)]

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

    # Run model on sample data and print results
    print_cls_results(fw_out[0], co_out[0], use_1k_labels=use_1k_labels)


variants = [
    ModelVariant.B0,
    ModelVariant.B1,
    ModelVariant.B2,
    ModelVariant.B3,
    ModelVariant.B4,
    ModelVariant.B5,
    ModelVariant.B6,
    ModelVariant.B7,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_efficientnet_torchvision(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.EFFICIENTNET,
        variant=variant,
        source=Source.TORCHVISION,
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

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    loader.print_cls_results(co_out)
