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


class Where0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where0_const_2", shape=(1, 256, 6, 20), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where0_const_2"))
        return where_output_1


class Where1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, where_input_0, where_input_1, where_input_2):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, where_input_2)
        return where_output_1


class Where2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where2_const_2", shape=(1, 256, 12, 40), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where2_const_2"))
        return where_output_1


class Where3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where3_const_2", shape=(1, 128, 12, 40), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where3_const_2"))
        return where_output_1


class Where4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where4_const_2", shape=(1, 128, 24, 80), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where4_const_2"))
        return where_output_1


class Where5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where5_const_2", shape=(1, 64, 24, 80), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where5_const_2"))
        return where_output_1


class Where6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where6_const_2", shape=(1, 64, 48, 160), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where6_const_2"))
        return where_output_1


class Where7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where7_const_2", shape=(1, 32, 48, 160), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where7_const_2"))
        return where_output_1


class Where8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where8_const_2", shape=(1, 32, 96, 320), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where8_const_2"))
        return where_output_1


class Where9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where9_const_2", shape=(1, 16, 96, 320), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where9_const_2"))
        return where_output_1


class Where10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where10_const_2", shape=(1, 16, 192, 640), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where10_const_2"))
        return where_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Where0,
        [((1, 256, 6, 20), torch.bool), ((1, 256, 6, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 256, 6, 20), torch.bool), ((1, 256, 6, 20), torch.bfloat16), ((1, 256, 6, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where2,
        [((1, 256, 12, 40), torch.bool), ((1, 256, 12, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 256, 12, 40), torch.bool), ((1, 256, 12, 40), torch.bfloat16), ((1, 256, 12, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where3,
        [((1, 128, 12, 40), torch.bool), ((1, 128, 12, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 128, 12, 40), torch.bool), ((1, 128, 12, 40), torch.bfloat16), ((1, 128, 12, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where4,
        [((1, 128, 24, 80), torch.bool), ((1, 128, 24, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 128, 24, 80), torch.bool), ((1, 128, 24, 80), torch.bfloat16), ((1, 128, 24, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where5,
        [((1, 64, 24, 80), torch.bool), ((1, 64, 24, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 64, 24, 80), torch.bool), ((1, 64, 24, 80), torch.bfloat16), ((1, 64, 24, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where6,
        [((1, 64, 48, 160), torch.bool), ((1, 64, 48, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 64, 48, 160), torch.bool), ((1, 64, 48, 160), torch.bfloat16), ((1, 64, 48, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where7,
        [((1, 32, 48, 160), torch.bool), ((1, 32, 48, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 32, 48, 160), torch.bool), ((1, 32, 48, 160), torch.bfloat16), ((1, 32, 48, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where8,
        [((1, 32, 96, 320), torch.bool), ((1, 32, 96, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 32, 96, 320), torch.bool), ((1, 32, 96, 320), torch.bfloat16), ((1, 32, 96, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where9,
        [((1, 16, 96, 320), torch.bool), ((1, 16, 96, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 16, 96, 320), torch.bool), ((1, 16, 96, 320), torch.bfloat16), ((1, 16, 96, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where10,
        [((1, 16, 192, 640), torch.bool), ((1, 16, 192, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 16, 192, 640), torch.bool), ((1, 16, 192, 640), torch.bfloat16), ((1, 16, 192, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Where1,
        [((1, 32, 480, 640), torch.bool), ((1, 32, 480, 640), torch.bfloat16), ((1, 32, 480, 640), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 64, 240, 320), torch.bool), ((1, 64, 240, 320), torch.bfloat16), ((1, 64, 240, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 32, 240, 320), torch.bool), ((1, 32, 240, 320), torch.bfloat16), ((1, 32, 240, 320), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 128, 120, 160), torch.bool), ((1, 128, 120, 160), torch.bfloat16), ((1, 128, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 64, 120, 160), torch.bool), ((1, 64, 120, 160), torch.bfloat16), ((1, 64, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 256, 60, 80), torch.bool), ((1, 256, 60, 80), torch.bfloat16), ((1, 256, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 128, 60, 80), torch.bool), ((1, 128, 60, 80), torch.bfloat16), ((1, 128, 60, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 512, 30, 40), torch.bool), ((1, 512, 30, 40), torch.bfloat16), ((1, 512, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 256, 30, 40), torch.bool), ((1, 256, 30, 40), torch.bfloat16), ((1, 256, 30, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 1024, 15, 20), torch.bool), ((1, 1024, 15, 20), torch.bfloat16), ((1, 1024, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Where1,
        [((1, 512, 15, 20), torch.bool), ((1, 512, 15, 20), torch.bfloat16), ((1, 512, 15, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Where")

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
