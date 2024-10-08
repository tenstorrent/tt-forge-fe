# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
from forge.verify.backend import verify_module
from forge import VerifyConfig
from forge.verify.config import TestKind
from transformers import (
    AutoImageProcessor,
    SegformerForImageClassification,
    SegformerConfig,
)

import os
import requests
import pytest
from PIL import Image


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


variants_img_classification = [
    "nvidia/mit-b0",
    "nvidia/mit-b1",
    "nvidia/mit-b2",
    "nvidia/mit-b3",
    "nvidia/mit-b4",
    "nvidia/mit-b5",
]


@pytest.mark.parametrize("variant", variants_img_classification)
def test_segformer_image_classification_pytorch(test_device, variant):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    os.environ["FORGE_RIBBON2"] = "1"
    os.environ["FORGE_DISABLE_PADDING_PASS"] = "1"
    pcc_value = 0.99

    if test_device.arch == forge.BackendDevice.Wormhole_B0:

        if variant in [
            "nvidia/mit-b1",
            "nvidia/mit-b2",
            "nvidia/mit-b3",
            "nvidia/mit-b4",
            "nvidia/mit-b5",
        ]:
            os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

        if variant == "nvidia/mit-b0" and test_device.devtype == forge.BackendType.Silicon:
            pcc_value = 0.97

    elif test_device.arch == forge.BackendDevice.Grayskull:

        if variant in ["nvidia/mit-b1"] and test_device.devtype == forge.BackendType.Silicon:
            pcc_value = 0.97

    # Set model configurations
    config = SegformerConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = SegformerConfig(**config_dict)

    # Load the model from HuggingFace
    model = SegformerForImageClassification.from_pretrained(variant, config=config)
    model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)

    # Create Forge module from PyTorch model
    tt_model = forge.PyTorchModule("pt_" + str(variant.split("/")[-1].replace("-", "_")), model)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(pixel_values.shape,)],
        inputs=[(pixel_values,)],
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_forge_codegen_vs_framework=True,
            verify_tvm_compile=True,
            pcc=pcc_value,
        ),
    )
