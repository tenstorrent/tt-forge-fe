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


class Add0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, add_input_0, add_input_1):
        add_output_1 = forge.op.Add("", add_input_0, add_input_1)
        return add_output_1


class Add1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add1.weight_1",
            forge.Parameter(
                *(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add1.weight_1"))
        return add_output_1


class Add2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add2.weight_1",
            forge.Parameter(
                *(8192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add2.weight_1"))
        return add_output_1


class Add3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add3.weight_1",
            forge.Parameter(*(2,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add3.weight_1"))
        return add_output_1


class Add4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add4.weight_1",
            forge.Parameter(
                *(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add4.weight_1"))
        return add_output_1


class Add5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add5.weight_1",
            forge.Parameter(
                *(3072,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add5.weight_1"))
        return add_output_1


class Add6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add6.weight_1",
            forge.Parameter(
                *(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add6.weight_1"))
        return add_output_1


class Add7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add7.weight_1",
            forge.Parameter(
                *(30000,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add7.weight_1"))
        return add_output_1


class Add8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add8.weight_1",
            forge.Parameter(
                *(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add8.weight_1"))
        return add_output_1


class Add9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add9.weight_1",
            forge.Parameter(
                *(16384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add9.weight_1"))
        return add_output_1


class Add10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add10.weight_1",
            forge.Parameter(
                *(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32, use_random_value=True
            ),
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add10.weight_1"))
        return add_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Add0,
        [((1, 128, 128), torch.float32), ((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_large_v1_token_cls",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add1,
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
        Add0,
        [((1, 16, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
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
        Add0,
        [((1, 128, 2048), torch.float32), ((1, 128, 2048), torch.float32)],
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
        Add2,
        [((1, 128, 8192), torch.float32)],
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
        Add3,
        [((1, 128, 2), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v2_token_cls",
                "pt_albert_base_v1_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_base_v2_token_cls",
                "pt_albert_xlarge_v2_token_cls",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add4,
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
        Add0,
        [((1, 12, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
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
        Add0,
        [((1, 128, 768), torch.float32), ((1, 128, 768), torch.float32)],
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
        Add5,
        [((1, 128, 3072), torch.float32)],
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
        Add6,
        [((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add7,
        [((1, 128, 30000), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_masked_lm",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_base_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add8,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_large_v1_token_cls",
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_albert_xxlarge_v1_masked_lm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Add0,
        [((1, 64, 128, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
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
        Add0,
        [((1, 128, 4096), torch.float32), ((1, 128, 4096), torch.float32)],
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
        Add9,
        [((1, 128, 16384), torch.float32)],
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
        Add10,
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
        Add0,
        [((1, 128, 1024), torch.float32), ((1, 128, 1024), torch.float32)],
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
