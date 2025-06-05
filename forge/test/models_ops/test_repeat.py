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


class Repeat0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeat_input_0):
        repeat_output_1 = forge.op.Repeat("", repeat_input_0, repeats=[1, 1, 1])
        return repeat_output_1


class Repeat1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeat_input_0):
        repeat_output_1 = forge.op.Repeat("", repeat_input_0, repeats=[1, 100, 1, 1, 1])
        return repeat_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Repeat0,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "pt_detr_facebook_detr_resnet_50_obj_det_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "[1, 1, 1]"},
        },
    ),
    pytest.param(
        (
            Repeat1,
            [((1, 1, 32, 107, 160), torch.float32)],
            {
                "model_names": [
                    "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                    "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                ],
                "pcc": 0.99,
                "args": {"repeats": "[1, 100, 1, 1, 1]"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Repeat1,
            [((1, 1, 64, 54, 80), torch.float32)],
            {
                "model_names": [
                    "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                    "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                ],
                "pcc": 0.99,
                "args": {"repeats": "[1, 100, 1, 1, 1]"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Repeat1,
            [((1, 1, 128, 27, 40), torch.float32)],
            {
                "model_names": [
                    "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                    "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                ],
                "pcc": 0.99,
                "args": {"repeats": "[1, 100, 1, 1, 1]"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Repeat1,
            [((1, 1, 256, 14, 20), torch.float32)],
            {
                "model_names": [
                    "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                    "pt_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                ],
                "pcc": 0.99,
                "args": {"repeats": "[1, 100, 1, 1, 1]"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Repeat")

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

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
