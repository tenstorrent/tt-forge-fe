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
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.config import VerifyConfig
import pytest


class Reshape0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 2048))
        return reshape_output_1


class Reshape1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 128))
        return reshape_output_1


class Reshape2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048))
        return reshape_output_1


class Reshape3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 128))
        return reshape_output_1


class Reshape4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 128))
        return reshape_output_1


class Reshape5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 2048, 1))
        return reshape_output_1


class Reshape6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 768))
        return reshape_output_1


class Reshape7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 12, 64))
        return reshape_output_1


class Reshape8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768))
        return reshape_output_1


class Reshape9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 64))
        return reshape_output_1


class Reshape10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 128))
        return reshape_output_1


class Reshape11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 128, 128))
        return reshape_output_1


class Reshape12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 64, 128))
        return reshape_output_1


class Reshape13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 128, 64))
        return reshape_output_1


class Reshape14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 768, 1))
        return reshape_output_1


class Reshape15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 4096))
        return reshape_output_1


class Reshape16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 64, 64))
        return reshape_output_1


class Reshape17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096))
        return reshape_output_1


class Reshape18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 64))
        return reshape_output_1


class Reshape19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 128))
        return reshape_output_1


class Reshape20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 128, 128))
        return reshape_output_1


class Reshape21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(64, 64, 128))
        return reshape_output_1


class Reshape22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 64, 128, 64))
        return reshape_output_1


class Reshape23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 4096, 1))
        return reshape_output_1


class Reshape24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(128, 1024))
        return reshape_output_1


class Reshape25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 16, 64))
        return reshape_output_1


class Reshape26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024))
        return reshape_output_1


class Reshape27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 128, 64))
        return reshape_output_1


class Reshape28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(16, 64, 128))
        return reshape_output_1


class Reshape29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 16, 128, 64))
        return reshape_output_1


class Reshape30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 128, 1024, 1))
        return reshape_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reshape0,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape1,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape2,
        [((128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape3,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape4,
        [((16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape5,
        [((1, 128, 16, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape6,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape7,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape8,
        [((128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape9,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape10,
        [((12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape11,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape12,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape13,
        [((12, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape14,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape15,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape16,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape17,
        [((128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape18,
        [((1, 64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape19,
        [((64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape20,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape21,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape22,
        [((64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape23,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape24,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape25,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape26,
        [((128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape27,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape28,
        [((1, 16, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape29,
        [((16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reshape30,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes
    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
