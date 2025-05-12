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


class Pad0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, pad_input_0):
        pad_output_1 = forge.op.Pad("", pad_input_0, pad=(0, 0, 2, 2), mode="constant", channel_last=True, value=0.0)
        return pad_output_1


class Pad1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, pad_input_0):
        pad_output_1 = forge.op.Pad("", pad_input_0, pad=(1, 1, 1, 1), mode="reflect", channel_last=False, value=0.0)
        return pad_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Pad0,
        [((1, 1, 96, 54, 54), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"pad": "(0, 0, 2, 2)", "mode": '"constant"', "channel_last": "True", "value": "0.0"},
        },
    ),
    (
        Pad0,
        [((1, 1, 256, 27, 27), torch.float32)],
        {
            "model_names": ["pt_alexnet_base_img_cls_osmr"],
            "pcc": 0.99,
            "args": {"pad": "(0, 0, 2, 2)", "mode": '"constant"', "channel_last": "True", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 512, 10, 32), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 512, 20, 64), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 256, 20, 64), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 256, 40, 128), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 128, 40, 128), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 128, 80, 256), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 64, 80, 256), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 96, 160, 512), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 32, 160, 512), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 16, 320, 1024), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 512, 6, 20), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 512, 12, 40), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 256, 12, 40), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 256, 24, 80), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 128, 24, 80), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 128, 48, 160), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 64, 48, 160), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 96, 96, 320), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 32, 96, 320), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
    (
        Pad1,
        [((1, 16, 192, 640), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"pad": "(1, 1, 1, 1)", "mode": '"reflect"', "channel_last": "False", "value": "0.0"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("Pad")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "args":
            forge_property_recorder.record_forge_op_args(metadata_value)
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

    forge_property_recorder.record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
