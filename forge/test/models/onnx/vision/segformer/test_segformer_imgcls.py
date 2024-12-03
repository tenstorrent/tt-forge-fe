# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
from forge.verify.backend import verify_module
from forge import DepricatedVerifyConfig
from forge.verify.config import TestKind
from transformers import AutoImageProcessor
import os
import pytest
import requests
from PIL import Image
import onnx


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


@pytest.mark.skip(reason="Not supported")
@pytest.mark.parametrize("variant", variants_img_classification)
@pytest.mark.nightly
def test_segformer_image_classification_onnx(test_device, variant):

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

    # Load the sample image
    pixel_values = get_sample_data(variant)

    onnx_model_path = (
        "third_party/confidential_customer_models/generated/files/"
        + str(variant).split("/")[-1].replace("-", "_")
        + ".onnx"
    )
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    tt_model = forge.OnnxModule(str(variant).split("/")[-1].replace("-", "_"), model, onnx_model_path)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(pixel_values.shape,)],
        inputs=[(pixel_values,)],
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_forge_codegen_vs_framework=True,
            verify_tvm_compile=True,
            pcc=pcc_value,
        ),
    )
