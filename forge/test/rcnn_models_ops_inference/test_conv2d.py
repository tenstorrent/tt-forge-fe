# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.verify import verify, verify_backward
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
            forge.Parameter(*(64, 3, 11, 11), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d0.weight_1"),
            stride=[4, 4],
            padding=[2, 2, 2, 2],
            dilation=1,
            groups=1,
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d1.weight_1",
            forge.Parameter(*(192, 64, 5, 5), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
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
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d2.weight_1",
            forge.Parameter(*(384, 192, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
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
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d3.weight_1",
            forge.Parameter(*(256, 384, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
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
            channel_last=False,
        )
        return conv2d_output_1


class Conv2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d4.weight_1",
            forge.Parameter(*(256, 256, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
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
            channel_last=False,
        )
        return conv2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Conv2D0,
        [((1, 3, 227, 227), torch.bfloat16)],
        {
            "model_names": ["pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[4, 4]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 64, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[2, 2, 2, 2]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D2,
        [((1, 192, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D3,
        [((1, 384, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
    (
        Conv2D4,
        [((1, 256, 13, 13), torch.bfloat16)],
        {
            "model_names": ["pt_rcnn_base_obj_det_torchvision_rect_0"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "False",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Conv2d")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc"]:
            continue
        if metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    enable_training = False
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad=enable_training)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            max_int=max_int,
            requires_grad=enable_training,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            max_int=max_int,
            requires_grad=enable_training,
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=enable_training)

    fw_out, co_out = verify(
        inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )
