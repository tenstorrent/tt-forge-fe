# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
from PIL import Image
import requests
from torchvision import transforms
import os
import pytest
from forge.verify.backend import verify_module
from forge.verify.config import TestKind
from forge import VerifyConfig
import sys

sys.path.append("third_party/confidential_customer_models/cv_demos/retinanet/model/")
from model_implementation import Model
from forge._C.backend_api import BackendDevice


def img_preprocess():

    url = "https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg"
    pil_img = Image.open(requests.get(url, stream=True).raw)
    new_size = (640, 480)
    pil_img = pil_img.resize(new_size, resample=Image.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(pil_img)
    img = img.unsqueeze(0)
    return img


variants = [
    "retinanet_rn18fpn",
    "retinanet_rn34fpn",
    "retinanet_rn50fpn",
    "retinanet_rn101fpn",
    "retinanet_rn152fpn",
]


@pytest.mark.parametrize("variant", variants)
def test_retinanet(variant, test_device):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    os.environ["FORGE_DECOMPOSE_SIGMOID"] = "1"
    os.environ["FORGE_RIBBON2"] = "1"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "73728"

    if test_device.arch == BackendDevice.Wormhole_B0:

        if variant == "retinanet_rn18fpn":
            compiler_cfg.place_on_new_epoch("conv2d_357.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_322.dc.matmul.11", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("conv2d_300.dc.matmul.11", "grid_shape", (1, 1))

        elif variant == "retinanet_rn34fpn":
            compiler_cfg.place_on_new_epoch("conv2d_589.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_554.dc.matmul.11", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("conv2d_532.dc.matmul.11", "grid_shape", (1, 1))

        elif variant == "retinanet_rn50fpn":
            compiler_cfg.place_on_new_epoch("conv2d_826.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_791.dc.matmul.11", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("conv2d_769.dc.matmul.11", "grid_shape", (1, 1))

        elif variant == "retinanet_rn101fpn":
            compiler_cfg.place_on_new_epoch("conv2d_1557.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_1522.dc.matmul.11", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("conv2d_1500.dc.matmul.11", "grid_shape", (1, 1))

        elif variant == "retinanet_rn152fpn":
            compiler_cfg.place_on_new_epoch("conv2d_2288.dc.matmul.11")
            compiler_cfg.balancer_op_override("conv2d_2253.dc.matmul.11", "t_stream_shape", (1, 1))
            compiler_cfg.balancer_op_override("conv2d_2231.dc.matmul.11", "grid_shape", (1, 1))

    if test_device.arch == BackendDevice.Grayskull:
        # Temp mitigations for net2pipe errors, should be removed.
        #
        os.environ["FORGE_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "0"
        os.environ["FORGE_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "0"
        os.environ["FORGE_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "0"

        if variant == "retinanet_rn18fpn":
            compiler_cfg.balancer_op_override("conv2d_322.dc.matmul.11", "t_stream_shape", (1, 1))

        elif variant == "retinanet_rn34fpn":
            compiler_cfg.balancer_op_override("conv2d_554.dc.matmul.11", "t_stream_shape", (1, 1))

        elif variant == "retinanet_rn50fpn":
            compiler_cfg.balancer_op_override("conv2d_791.dc.matmul.11", "t_stream_shape", (1, 1))

        elif variant == "retinanet_rn101fpn":
            compiler_cfg.balancer_op_override("conv2d_1522.dc.matmul.11", "t_stream_shape", (1, 1))

        elif variant == "retinanet_rn152fpn":
            compiler_cfg.balancer_op_override("conv2d_2253.dc.matmul.11", "t_stream_shape", (1, 1))

    # Prepare model

    checkpoint_path = f"third_party/confidential_customer_models/cv_demos/retinanet/weights/{variant}.pth"
    model = Model.load(checkpoint_path)
    model.eval()
    tt_model = forge.PyTorchModule(f"pt_{variant}", model)

    # Prepare input
    input_batch = img_preprocess()

    # Inference
    verify_module(
        tt_model,
        input_shapes=([input_batch.shape]),
        inputs=([input_batch]),
        verify_cfg=VerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
        ),
    )
