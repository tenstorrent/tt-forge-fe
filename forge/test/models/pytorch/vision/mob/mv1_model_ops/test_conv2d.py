# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

import forge
import forge.op
from forge import ForgeModule, Tensor, compile
from forge.forge_property_utils import (
    record_forge_op_args,
    record_forge_op_name,
    record_op_model_names,
    record_single_op_operands_info,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify


class Conv2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d0.weight_1",
            forge.Parameter(*(32, 3, 3, 3), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d0.weight_1"),
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=32,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d2.weight_1",
            forge.Parameter(*(64, 32, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d2.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=64,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d4.weight_1",
            forge.Parameter(*(128, 64, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d4.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=128,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=128,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d7.weight_1",
            forge.Parameter(*(128, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d7.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
            forge.Parameter(*(256, 128, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d8.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=256,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=256,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d11.weight_1",
            forge.Parameter(*(256, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d11.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d12.weight_1",
            forge.Parameter(*(512, 256, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d12.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=512,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[2, 2],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=512,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d15.weight_1",
            forge.Parameter(*(512, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d15.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d16.weight_1",
            forge.Parameter(*(1024, 512, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d16.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=1,
            groups=1,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, conv2d_input_0, conv2d_input_1):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            conv2d_input_1,
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=1,
            groups=1024,
            channel_last=0,
        )
        return conv2d_output_1


class Conv2D18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2d18.weight_1",
            forge.Parameter(*(1024, 1024, 1, 1), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, conv2d_input_0):
        conv2d_output_1 = forge.op.Conv2d(
            "",
            conv2d_input_0,
            self.get_parameter("conv2d18.weight_1"),
            stride=[1, 1],
            padding=[0, 0, 0, 0],
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
        [((1, 3, 224, 224), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D1,
        [((1, 32, 112, 112), torch.bfloat16), ((32, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "32",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D2,
        [((1, 32, 112, 112), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D3,
        [((1, 64, 112, 112), torch.bfloat16), ((64, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "64",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D4,
        [((1, 64, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D5,
        [((1, 128, 56, 56), torch.bfloat16), ((128, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "128",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D6,
        [((1, 128, 56, 56), torch.bfloat16), ((128, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "128",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D7,
        [((1, 128, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D8,
        [((1, 128, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D9,
        [((1, 256, 28, 28), torch.bfloat16), ((256, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "256",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D10,
        [((1, 256, 28, 28), torch.bfloat16), ((256, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "256",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D11,
        [((1, 256, 28, 28), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D12,
        [((1, 256, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D13,
        [((1, 512, 14, 14), torch.bfloat16), ((512, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "512",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D14,
        [((1, 512, 14, 14), torch.bfloat16), ((512, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[2, 2]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "512",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D15,
        [((1, 512, 14, 14), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D16,
        [((1, 512, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D17,
        [((1, 1024, 7, 7), torch.bfloat16), ((1024, 1, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[1, 1, 1, 1]",
                "dilation": "1",
                "groups": "1024",
                "channel_last": "0",
            },
        },
    ),
    (
        Conv2D18,
        [((1, 1024, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {
                "stride": "[1, 1]",
                "padding": "[0, 0, 0, 0]",
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

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
