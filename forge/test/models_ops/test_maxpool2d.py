# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.verify import verify
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
from forge.forge_property_utils import (
    record_forge_op_name,
    record_op_model_names,
    record_forge_op_args,
    record_single_op_operands_info,
)
import pytest


class Maxpool2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[1, 1, 1, 1],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=2,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 2, 0, 2],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 0, 1, 1],
            dilation=1,
            ceil_mode=False,
            channel_last=1,
        )
        return maxpool2d_output_1


class Maxpool2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=5,
            stride=1,
            padding=[2, 2, 2, 2],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


class Maxpool2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, maxpool2d_input_0):
        maxpool2d_output_1 = forge.op.MaxPool2d(
            "",
            maxpool2d_input_0,
            kernel_size=3,
            stride=2,
            padding=[0, 0, 0, 0],
            dilation=1,
            ceil_mode=False,
            channel_last=0,
        )
        return maxpool2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Maxpool2D0,
        [((1, 64, 214, 320), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 32, 112, 112), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 128, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 2, 0, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 2, 0, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 128, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 384, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 2, 0, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D0,
        [((1, 64, 112, 112), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 2, 0, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 2, 0, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D2,
        [((1, 768, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 2, 0, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D3,
        [((1, 112, 112, 64), torch.float32)],
        {
            "model_names": ["jax_resnet_50_img_cls_hf"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 1, 1]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "1",
            },
        },
    ),
    (
        Maxpool2D4,
        [((1, 128, 20, 20), torch.float32)],
        {
            "model_names": ["onnx_yolov8_default_obj_det_github"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "5",
                "stride": "1",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 128, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D1,
        [((1, 256, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla34_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "2",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 64, 55, 55), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 192, 27, 27), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
    (
        Maxpool2D5,
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_names": ["pd_alexnet_base_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {
                "kernel_size": "3",
                "stride": "2",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "ceil_mode": "False",
                "channel_last": "0",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("MaxPool2d")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format, max_int=max_int
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format, max_int=max_int
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
