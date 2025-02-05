# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import forge
import onnx

import os
import requests
from PIL import Image
import pytest

from transformers import AutoImageProcessor

from forge.verify.backend import verify_module
from forge import DepricatedVerifyConfig
from forge.verify.config import TestKind


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Not supported")
@pytest.mark.parametrize(
    "model_name",
    [
        "deepmind/vision-perceiver-conv",
        "deepmind/vision-perceiver-learned",
        "deepmind/vision-perceiver-fourier",
    ],
)
@pytest.mark.nightly
def test_perceiver_for_image_classification_onnx(test_device, model_name):

    # Set Forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    verify_enabled = True

    onnx_model_path = (
        "third_party/confidential_customer_models/generated/files/"
        + str(model_name).split("/")[-1].replace("-", "_")
        + ".onnx"
    )

    # Sample Image
    pixel_values = get_sample_data(model_name)

    # Load the onnx model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # Create Forge module from Onnx model
    tt_model = forge.OnnxModule(
        str(model_name.split("/")[-1].replace("-", "_")) + "_onnx",
        onnx_model,
        onnx_model_path,
    )

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=(pixel_values.shape,),
        inputs=[(pixel_values,)],
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            enabled=verify_enabled,  # pcc drops in silicon devicetype
            pcc=0.96,
        ),
    )
