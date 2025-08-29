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
            "add1.weight_1", forge.Parameter(*(16,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add1.weight_1"))
        return add_output_1


class Add2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add2.weight_1", forge.Parameter(*(8,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add2.weight_1"))
        return add_output_1


class Add3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add3.weight_1", forge.Parameter(*(48,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add3.weight_1"))
        return add_output_1


class Add4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add4.weight_1", forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add4.weight_1"))
        return add_output_1


class Add5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add5.weight_1", forge.Parameter(*(24,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add5.weight_1"))
        return add_output_1


class Add6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add6.weight_1", forge.Parameter(*(144,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add6.weight_1"))
        return add_output_1


class Add7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add7.weight_1", forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add7.weight_1"))
        return add_output_1


class Add8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add8.weight_1", forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add8.weight_1"))
        return add_output_1


class Add9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add9.weight_1", forge.Parameter(*(56,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add9.weight_1"))
        return add_output_1


class Add10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add10.weight_1", forge.Parameter(*(336,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add10.weight_1"))
        return add_output_1


class Add11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add11.weight_1", forge.Parameter(*(112,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add11.weight_1"))
        return add_output_1


class Add12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add12.weight_1", forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add12.weight_1"))
        return add_output_1


class Add13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "add13.weight_1", forge.Parameter(*(1001,), requires_grad=True, dev_data_format=forge.DataFormat.Float16_b)
        )

    def forward(self, add_input_0):
        add_output_1 = forge.op.Add("", add_input_0, self.get_parameter("add13.weight_1"))
        return add_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Add0,
        [((16,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add1,
        [((16,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 16, 48, 48), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((8,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add2,
        [((8,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 8, 48, 48), torch.bfloat16), ((8, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((48,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add3,
        [((48,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 48, 48), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 24, 24), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 8, 24, 24), torch.bfloat16), ((8, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 8, 24, 24), torch.bfloat16), ((1, 8, 24, 24), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 48, 12, 12), torch.bfloat16), ((48, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 16, 12, 12), torch.bfloat16), ((16, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((96,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add4,
        [((96,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 12, 12), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 16, 12, 12), torch.bfloat16), ((1, 16, 12, 12), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 96, 6, 6), torch.bfloat16), ((96, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((24,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add5,
        [((24,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 6, 6), torch.bfloat16), ((24, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((144,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add6,
        [((144,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 144, 6, 6), torch.bfloat16), ((144, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 24, 6, 6), torch.bfloat16), ((1, 24, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((32,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add7,
        [((32,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 6, 6), torch.bfloat16), ((32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((192,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add8,
        [((192,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 6, 6), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 32, 6, 6), torch.bfloat16), ((1, 32, 6, 6), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 192, 3, 3), torch.bfloat16), ((192, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((56,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add9,
        [((56,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 56, 3, 3), torch.bfloat16), ((56, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((336,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add10,
        [((336,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 336, 3, 3), torch.bfloat16), ((336, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 56, 3, 3), torch.bfloat16), ((1, 56, 3, 3), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((112,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add11,
        [((112,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 112, 3, 3), torch.bfloat16), ((112, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1280,), torch.bfloat16), ((1,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add12,
        [((1280,), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add0,
        [((1, 1280, 3, 3), torch.bfloat16), ((1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Add13,
        [((1, 1001), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
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
