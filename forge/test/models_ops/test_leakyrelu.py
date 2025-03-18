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


class Leakyrelu0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, leakyrelu_input_0):
        leakyrelu_output_1 = forge.op.LeakyRelu("", leakyrelu_input_0, alpha=0.10000000000000001)
        return leakyrelu_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Leakyrelu0,
        [((1, 32, 640, 640), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 64, 320, 320), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 32, 320, 320), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 128, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 64, 160, 160), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 256, 80, 80), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 128, 80, 80), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 512, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 256, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 1024, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 512, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 256, 20, 20), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
    (
        Leakyrelu0,
        [((1, 128, 40, 40), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_darknet_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"alpha": "0.10000000000000001"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder.record_op_name("LeakyRelu")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        forge_property_recorder("tags." + str(metadata_name), metadata_value)

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

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
