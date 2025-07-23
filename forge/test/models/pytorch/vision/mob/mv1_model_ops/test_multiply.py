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


class Multiply0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply0.weight_1",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply0.weight_1"))
        return multiply_output_1


class Multiply1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, multiply_input_0, multiply_input_1):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, multiply_input_1)
        return multiply_output_1


class Multiply2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply2.weight_1",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply2.weight_1"))
        return multiply_output_1


class Multiply3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply3.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply3.weight_1"))
        return multiply_output_1


class Multiply4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply4.weight_1",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply4.weight_1"))
        return multiply_output_1


class Multiply5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply5.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply5.weight_1"))
        return multiply_output_1


class Multiply6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "multiply6.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b),
        )

    def forward(self, multiply_input_0):
        multiply_output_1 = forge.op.Multiply("", multiply_input_0, self.get_parameter("multiply6.weight_1"))
        return multiply_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Multiply0,
        [((32,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 32, 112, 112), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((32,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((32,), torch.bfloat16), ((32,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply2,
        [((64,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 64, 112, 112), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((64,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((64,), torch.bfloat16), ((64,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 64, 56, 56), torch.bfloat16), ((64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply3,
        [((128,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 128, 56, 56), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((128,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((128,), torch.bfloat16), ((128,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 128, 28, 28), torch.bfloat16), ((128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply4,
        [((256,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 256, 28, 28), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((256,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((256,), torch.bfloat16), ((256,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 256, 14, 14), torch.bfloat16), ((256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply5,
        [((512,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 512, 14, 14), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((512,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((512,), torch.bfloat16), ((512,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 512, 7, 7), torch.bfloat16), ((512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply6,
        [((1024,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1, 1024, 7, 7), torch.bfloat16), ((1024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1024,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Multiply1,
        [((1024,), torch.bfloat16), ((1024,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Multiply")

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
