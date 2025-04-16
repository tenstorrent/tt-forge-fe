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
import pytest


class Avgpool3D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, avgpool3d_input_0):
        avgpool3d_output_1 = forge.op.AvgPool3d(
            "",
            avgpool3d_input_0,
            kernel_size=[5, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0, 0, 0, 0],
            ceil_mode=False,
            count_include_pad=True,
            channel_last=0,
        )
        return avgpool3d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Avgpool3D0,
            [((1, 1, 100, 54, 54), torch.float32)],
            {
                "model_name": ["pt_alexnet_base_img_cls_osmr"],
                "pcc": 0.99,
                "op_params": {
                    "kernel_size": "[5, 1, 1]",
                    "stride": "[1, 1, 1]",
                    "padding": "[0, 0, 0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Avgpool3D0,
            [((1, 1, 260, 27, 27), torch.float32)],
            {
                "model_name": ["pt_alexnet_base_img_cls_osmr"],
                "pcc": 0.99,
                "op_params": {
                    "kernel_size": "[5, 1, 1]",
                    "stride": "[1, 1, 1]",
                    "padding": "[0, 0, 0, 0, 0, 0]",
                    "ceil_mode": "False",
                    "count_include_pad": "True",
                    "channel_last": "0",
                },
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("AvgPool3d")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_name":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "op_params":
            forge_property_recorder.record_forge_op_args(metadata_value)
        else:
            logger.warning("no utility function in forge property handler")

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

    forge_property_recorder.record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
