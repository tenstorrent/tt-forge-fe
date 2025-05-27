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


class Conv2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d0.weight_1",
            forge.Parameter(*(96, 3, 11, 11), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d0.weight_1"),
            stride=[4, 4],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1.weight_1",
            forge.Parameter(*(256, 96, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d1.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d2.weight_1",
            forge.Parameter(*(384, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d2.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d3.weight_1",
            forge.Parameter(*(384, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d3.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d4.weight_1",
            forge.Parameter(*(256, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d4.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d5.weight_1",
            forge.Parameter(*(64, 3, 11, 11), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d5.weight_1"),
            stride=[4, 4],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d6.weight_1",
            forge.Parameter(*(192, 64, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d6.weight_1"),
            stride=[1, 1],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d7.weight_1",
            forge.Parameter(*(384, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d7.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d8.weight_1",
            forge.Parameter(*(256, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d8.weight_1"),
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Conv2D0,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[4, 4]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 96, 27, 27), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D2,
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D3,
        [((1, 384, 13, 13), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D4,
        [((1, 384, 13, 13), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr", "pt_alexnet_base_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D5,
        [((1, 3, 224, 224), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[4, 4]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D6,
        [((1, 64, 27, 27), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D7,
        [((1, 192, 13, 13), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D8,
        [((1, 256, 13, 13), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Conv2d")

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
