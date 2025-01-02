# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.compare import compare_with_golden
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
import pytest


class Leakyrelu0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, leakyrelu_input_0):
        leakyrelu_output_1 = forge.op.LeakyRelu("", leakyrelu_input_0, alpha=0.10000000000000001)
        return leakyrelu_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Leakyrelu0, [((1, 32, 640, 640), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 64, 320, 320), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 32, 320, 320), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 128, 160, 160), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 64, 160, 160), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 256, 80, 80), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 128, 80, 80), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 512, 40, 40), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 256, 40, 40), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 1024, 20, 20), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 512, 20, 20), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 256, 20, 20), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
    (Leakyrelu0, [((1, 128, 40, 40), torch.float32)], {"model_name": ["pt_yolox_darknet"]}),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
