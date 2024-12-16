# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import onnx
import pytest
import requests
from PIL import Image
from transformers import AutoImageProcessor

import forge
from forge import DepricatedVerifyConfig
from forge.verify.backend import verify_module
from forge.verify.config import TestKind


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


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
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    os.environ["FORGE_RIBBON2"] = "1"
    verify_enabled = True

    if test_device.arch == forge.BackendDevice.Wormhole_B0:

        if model_name == "deepmind/vision-perceiver-learned":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{105*1024}"

        elif model_name == "deepmind/vision-perceiver-conv":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{10*1024}"
            compiler_cfg.balancer_op_override("multiply_19", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("multiply_142", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("multiply_3103", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("multiply_3123", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("multiply_2745", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("multiply_2934", "t_stream_shape", (1, 1))

        elif model_name == "deepmind/vision-perceiver-fourier":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{101*1024}"

    elif test_device.arch == forge.BackendDevice.Grayskull:

        if test_device.devtype == forge.BackendType.Silicon:
            verify_enabled = False

        if model_name == "deepmind/vision-perceiver-learned":
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{101*1024}"

        elif model_name == "deepmind/vision-perceiver-fourier":
            os.environ["FORGE_DISABLE_PADDING_PASS"] = "1"
            compiler_cfg.place_on_new_epoch("hslice_50.dc.sparse_matmul.2.lc2")
            compiler_cfg.place_on_new_epoch("matmul_47")
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{101*1024}"
            compiler_cfg.balancer_op_override("hslice_50.dc.sparse_matmul.2.lc2", "t_stream_shape", (1, 7))

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
