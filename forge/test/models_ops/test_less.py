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


class Less0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less0_const_1", shape=(1, 256, 6, 20), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less0_const_1"))
        return less_output_1


class Less1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less1_const_1", shape=(1, 256, 12, 40), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less1_const_1"))
        return less_output_1


class Less2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less2_const_1", shape=(1, 128, 12, 40), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less2_const_1"))
        return less_output_1


class Less3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less3_const_1", shape=(1, 128, 24, 80), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less3_const_1"))
        return less_output_1


class Less4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less4_const_1", shape=(1, 64, 24, 80), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less4_const_1"))
        return less_output_1


class Less5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less5_const_1", shape=(1, 64, 48, 160), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less5_const_1"))
        return less_output_1


class Less6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less6_const_1", shape=(1, 32, 48, 160), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less6_const_1"))
        return less_output_1


class Less7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less7_const_1", shape=(1, 32, 96, 320), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less7_const_1"))
        return less_output_1


class Less8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less8_const_1", shape=(1, 16, 96, 320), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less8_const_1"))
        return less_output_1


class Less9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less9_const_1", shape=(1, 16, 192, 640), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less9_const_1"))
        return less_output_1


class Less10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less10_const_1", shape=(1, 256, 10, 32), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less10_const_1"))
        return less_output_1


class Less11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less11_const_1", shape=(1, 256, 20, 64), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less11_const_1"))
        return less_output_1


class Less12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less12_const_1", shape=(1, 128, 20, 64), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less12_const_1"))
        return less_output_1


class Less13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less13_const_1", shape=(1, 128, 40, 128), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less13_const_1"))
        return less_output_1


class Less14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less14_const_1", shape=(1, 64, 40, 128), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less14_const_1"))
        return less_output_1


class Less15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less15_const_1", shape=(1, 64, 80, 256), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less15_const_1"))
        return less_output_1


class Less16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less16_const_1", shape=(1, 32, 80, 256), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less16_const_1"))
        return less_output_1


class Less17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less17_const_1", shape=(1, 32, 160, 512), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less17_const_1"))
        return less_output_1


class Less18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less18_const_1", shape=(1, 16, 160, 512), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less18_const_1"))
        return less_output_1


class Less19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("less19_const_1", shape=(1, 16, 320, 1024), dtype=torch.float32)

    def forward(self, less_input_0):
        less_output_1 = forge.op.Less("", less_input_0, self.get_constant("less19_const_1"))
        return less_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Less0,
        [((1, 256, 6, 20), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less1,
        [((1, 256, 12, 40), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less2,
        [((1, 128, 12, 40), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less3,
        [((1, 128, 24, 80), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less4,
        [((1, 64, 24, 80), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less5,
        [((1, 64, 48, 160), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less6,
        [((1, 32, 48, 160), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less7,
        [((1, 32, 96, 320), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less8,
        [((1, 16, 96, 320), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less9,
        [((1, 16, 192, 640), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less10,
        [((1, 256, 10, 32), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less11,
        [((1, 256, 20, 64), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less12,
        [((1, 128, 20, 64), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less13,
        [((1, 128, 40, 128), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less14,
        [((1, 64, 40, 128), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less15,
        [((1, 64, 80, 256), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less16,
        [((1, 32, 80, 256), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less17,
        [((1, 32, 160, 512), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less18,
        [((1, 16, 160, 512), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Less19,
        [((1, 16, 320, 1024), torch.float32)],
        {
            "model_name": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("Less")

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
