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


class Conv2Dtranspose0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose0.weight_1",
            forge.Parameter(*(4, 16, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose0.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose1.weight_1",
            forge.Parameter(*(16, 1, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose1.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose2.weight_1",
            forge.Parameter(*(64, 1, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose2.weight_1"),
            stride=2,
            padding=1,
            dilation=1,
            groups=64,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose3.weight_1",
            forge.Parameter(*(128, 1, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose3.weight_1"),
            stride=2,
            padding=1,
            dilation=1,
            groups=128,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose4.weight_1",
            forge.Parameter(*(256, 1, 4, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose4.weight_1"),
            stride=2,
            padding=1,
            dilation=1,
            groups=256,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose5.weight_1",
            forge.Parameter(*(512, 256, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose5.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose6.weight_1",
            forge.Parameter(*(256, 128, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose6.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose7.weight_1",
            forge.Parameter(*(128, 64, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose7.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose8.weight_1",
            forge.Parameter(*(64, 32, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose8.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


class Conv2Dtranspose9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "conv2dtranspose9.weight_1",
            forge.Parameter(*(1024, 512, 2, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, conv2dtranspose_input_0):
        conv2dtranspose_output_1 = forge.op.Conv2dTranspose(
            "",
            conv2dtranspose_input_0,
            self.get_parameter("conv2dtranspose9.weight_1"),
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            channel_last=0,
            output_padding=[0, 0],
        )
        return conv2dtranspose_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Conv2Dtranspose0,
        [((1, 4, 7, 7), torch.float32)],
        {
            "model_name": ["pt_autoencoder_conv_img_enc_github"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    (
        Conv2Dtranspose1,
        [((1, 16, 14, 14), torch.float32)],
        {
            "model_name": ["pt_autoencoder_conv_img_enc_github"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    pytest.param(
        (
            Conv2Dtranspose2,
            [((1, 64, 28, 28), torch.float32)],
            {
                "model_name": ["pt_monodle_base_obj_det_torchvision"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "1",
                    "dilation": "1",
                    "groups": "64",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[pytest.mark.skip(reason="tt_forge_signal_handler Floating point exception")],
    ),
    pytest.param(
        (
            Conv2Dtranspose3,
            [((1, 128, 14, 14), torch.float32)],
            {
                "model_name": ["pt_monodle_base_obj_det_torchvision"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "1",
                    "dilation": "1",
                    "groups": "128",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[pytest.mark.skip(reason="tt_forge_signal_handler Floating point exception")],
    ),
    pytest.param(
        (
            Conv2Dtranspose4,
            [((1, 256, 7, 7), torch.float32)],
            {
                "model_name": ["pt_monodle_base_obj_det_torchvision"],
                "pcc": 0.99,
                "op_params": {
                    "stride": "2",
                    "padding": "1",
                    "dilation": "1",
                    "groups": "256",
                    "channel_last": "0",
                    "output_padding": "[0, 0]",
                },
            },
        ),
        marks=[pytest.mark.skip(reason="tt_forge_signal_handler Floating point exception")],
    ),
    (
        Conv2Dtranspose5,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_name": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    (
        Conv2Dtranspose6,
        [((1, 256, 32, 32), torch.float32)],
        {
            "model_name": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    (
        Conv2Dtranspose7,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_name": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    (
        Conv2Dtranspose8,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_name": ["pt_unet_base_img_seg_torchhub"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    (
        Conv2Dtranspose9,
        [((1, 1024, 14, 14), torch.float32)],
        {
            "model_name": ["pt_unet_carvana_base_img_seg_github"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    (
        Conv2Dtranspose5,
        [((1, 512, 28, 28), torch.float32)],
        {
            "model_name": ["pt_unet_carvana_base_img_seg_github"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    (
        Conv2Dtranspose6,
        [((1, 256, 56, 56), torch.float32)],
        {
            "model_name": ["pt_unet_carvana_base_img_seg_github"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
    (
        Conv2Dtranspose7,
        [((1, 128, 112, 112), torch.float32)],
        {
            "model_name": ["pt_unet_carvana_base_img_seg_github"],
            "pcc": 0.99,
            "op_params": {
                "stride": "2",
                "padding": "0",
                "dilation": "1",
                "groups": "1",
                "channel_last": "0",
                "output_padding": "[0, 0]",
            },
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Conv2dTranspose")

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
