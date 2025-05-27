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


class Add0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, add_input_0, add_input_1):
        add_output_1 = forge.op.Add("", add_input_0, add_input_1)
        return add_output_1


class Add1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("add1_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_constant("add1_const_1"))
        return add_output_1


class Add2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add2.weight_1", forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add2.weight_1"))
        return add_output_1


class Add3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add3.weight_1", forge.Parameter(*(1000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add3.weight_1"))
        return add_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Add0,
        [((1, 96, 54, 54), torch.float32), ((96, 1, 1), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99},
    ),
    (Add1, [((1, 96, 54, 54), torch.float32)], {"model_names": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99}),
    (
        Add0,
        [((1, 256, 27, 27), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99},
    ),
    (Add1, [((1, 256, 27, 27), torch.float32)], {"model_names": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99}),
    (
        Add0,
        [((1, 384, 13, 13), torch.float32), ((384, 1, 1), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_osmr", "pt_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 256, 13, 13), torch.float32), ((256, 1, 1), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_osmr", "pt_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add2,
        [((1, 4096), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_osmr", "pt_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add3,
        [((1, 1000), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_osmr", "pt_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 64, 55, 55), torch.float32), ((64, 1, 1), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Add0,
        [((1, 192, 27, 27), torch.float32), ((192, 1, 1), torch.float32)],
        {"model_names": ["pt_alexnet_base_img_cls_torchhub"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Add")

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
