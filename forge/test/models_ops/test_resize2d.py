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


class Resize2D0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[16, 16], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[64, 64], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[128, 128], method="linear", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[56, 56], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[28, 28], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[14, 14], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[28, 28], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[20, 64], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[40, 128], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[80, 256], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[160, 512], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[320, 1024], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[12, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[24, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[48, 160], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[96, 320], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[192, 640], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[30, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[60, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[112, 112], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[224, 224], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[56, 56], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[112, 112], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[224, 224], method="linear", align_corners=True, channel_last=0
        )
        return resize2d_output_1


class Resize2D24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[40, 40], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[80, 80], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[30, 30], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[60, 60], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[20, 20], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[160, 160], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[26, 26], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


class Resize2D31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, resize2d_input_0):
        resize2d_output_1 = forge.op.Resize2d(
            "", resize2d_input_0, sizes=[52, 52], method="nearest_neighbor", align_corners=False, channel_last=0
        )
        return resize2d_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Resize2D0, [((1, 256, 8, 8), torch.float32)]),
    (Resize2D1, [((1, 256, 16, 16), torch.float32)]),
    (Resize2D2, [((1, 256, 16, 16), torch.float32)]),
    (Resize2D3, [((1, 18, 28, 28), torch.float32)]),
    (Resize2D3, [((1, 18, 14, 14), torch.float32)]),
    (Resize2D4, [((1, 36, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 18, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 36, 7, 7), torch.float32)]),
    (Resize2D5, [((1, 72, 7, 7), torch.float32)]),
    (Resize2D3, [((1, 40, 28, 28), torch.float32)]),
    (Resize2D3, [((1, 40, 14, 14), torch.float32)]),
    (Resize2D4, [((1, 80, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 40, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 80, 7, 7), torch.float32)]),
    (Resize2D5, [((1, 160, 7, 7), torch.float32)]),
    (Resize2D3, [((1, 64, 28, 28), torch.float32)]),
    (Resize2D3, [((1, 64, 14, 14), torch.float32)]),
    (Resize2D4, [((1, 64, 14, 14), torch.float32)]),
    (Resize2D4, [((1, 128, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 64, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 64, 7, 7), torch.float32)]),
    (Resize2D5, [((1, 64, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 128, 7, 7), torch.float32)]),
    (Resize2D5, [((1, 128, 7, 7), torch.float32)]),
    (Resize2D5, [((1, 256, 7, 7), torch.float32)]),
    (Resize2D3, [((1, 32, 28, 28), torch.float32)]),
    (Resize2D3, [((1, 32, 14, 14), torch.float32)]),
    (Resize2D4, [((1, 32, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 32, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 32, 7, 7), torch.float32)]),
    (Resize2D3, [((1, 16, 28, 28), torch.float32)]),
    (Resize2D3, [((1, 16, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 16, 7, 7), torch.float32)]),
    (Resize2D3, [((1, 44, 28, 28), torch.float32)]),
    (Resize2D3, [((1, 44, 14, 14), torch.float32)]),
    (Resize2D4, [((1, 88, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 44, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 88, 7, 7), torch.float32)]),
    (Resize2D5, [((1, 176, 7, 7), torch.float32)]),
    (Resize2D3, [((1, 48, 28, 28), torch.float32)]),
    (Resize2D3, [((1, 48, 14, 14), torch.float32)]),
    (Resize2D4, [((1, 96, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 48, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 96, 7, 7), torch.float32)]),
    (Resize2D5, [((1, 192, 7, 7), torch.float32)]),
    (Resize2D3, [((1, 30, 28, 28), torch.float32)]),
    (Resize2D3, [((1, 30, 14, 14), torch.float32)]),
    (Resize2D4, [((1, 60, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 30, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 60, 7, 7), torch.float32)]),
    (Resize2D5, [((1, 120, 7, 7), torch.float32)]),
    (Resize2D6, [((1, 256, 1, 1), torch.float32)]),
    (Resize2D7, [((1, 256, 10, 32), torch.float32)]),
    (Resize2D8, [((1, 128, 20, 64), torch.float32)]),
    (Resize2D9, [((1, 64, 40, 128), torch.float32)]),
    (Resize2D10, [((1, 32, 80, 256), torch.float32)]),
    (Resize2D11, [((1, 16, 160, 512), torch.float32)]),
    (Resize2D12, [((1, 256, 6, 20), torch.float32)]),
    (Resize2D13, [((1, 128, 12, 40), torch.float32)]),
    (Resize2D14, [((1, 64, 24, 80), torch.float32)]),
    (Resize2D15, [((1, 32, 48, 160), torch.float32)]),
    (Resize2D16, [((1, 16, 96, 320), torch.float32)]),
    (Resize2D17, [((1, 256, 15, 20), torch.float32)]),
    (Resize2D18, [((1, 256, 30, 40), torch.float32)]),
    (Resize2D2, [((1, 256, 32, 32), torch.float32)]),
    (Resize2D2, [((1, 256, 64, 64), torch.float32)]),
    (Resize2D2, [((1, 256, 128, 128), torch.float32)]),
    (Resize2D2, [((1, 768, 16, 16), torch.float32)]),
    (Resize2D2, [((1, 768, 32, 32), torch.float32)]),
    (Resize2D2, [((1, 768, 64, 64), torch.float32)]),
    (Resize2D2, [((1, 768, 128, 128), torch.float32)]),
    (Resize2D5, [((1, 2048, 7, 7), torch.float32)]),
    (Resize2D4, [((1, 256, 14, 14), torch.float32)]),
    (Resize2D3, [((1, 128, 28, 28), torch.float32)]),
    (Resize2D19, [((1, 64, 56, 56), torch.float32)]),
    (Resize2D20, [((1, 32, 112, 112), torch.float32)]),
    (Resize2D6, [((1, 512, 14, 14), torch.float32)]),
    (Resize2D21, [((1, 256, 28, 28), torch.float32)]),
    (Resize2D22, [((1, 128, 56, 56), torch.float32)]),
    (Resize2D23, [((1, 64, 112, 112), torch.float32)]),
    (Resize2D24, [((1, 640, 20, 20), torch.float32)]),
    (Resize2D25, [((1, 320, 40, 40), torch.float32)]),
    (Resize2D26, [((1, 384, 15, 15), torch.float32)]),
    (Resize2D27, [((1, 192, 30, 30), torch.float32)]),
    (Resize2D28, [((1, 640, 10, 10), torch.float32)]),
    (Resize2D24, [((1, 320, 20, 20), torch.float32)]),
    (Resize2D24, [((1, 256, 20, 20), torch.float32)]),
    (Resize2D25, [((1, 128, 40, 40), torch.float32)]),
    (Resize2D28, [((1, 512, 10, 10), torch.float32)]),
    (Resize2D24, [((1, 512, 20, 20), torch.float32)]),
    (Resize2D25, [((1, 256, 40, 40), torch.float32)]),
    (Resize2D29, [((1, 128, 80, 80), torch.float32)]),
    (Resize2D24, [((1, 128, 20, 20), torch.float32)]),
    (Resize2D25, [((1, 64, 40, 40), torch.float32)]),
    (Resize2D26, [((1, 256, 15, 15), torch.float32)]),
    (Resize2D27, [((1, 128, 30, 30), torch.float32)]),
    (Resize2D26, [((1, 640, 15, 15), torch.float32)]),
    (Resize2D27, [((1, 320, 30, 30), torch.float32)]),
    (Resize2D26, [((1, 128, 15, 15), torch.float32)]),
    (Resize2D27, [((1, 64, 30, 30), torch.float32)]),
    (Resize2D26, [((1, 512, 15, 15), torch.float32)]),
    (Resize2D27, [((1, 256, 30, 30), torch.float32)]),
    (Resize2D24, [((1, 384, 20, 20), torch.float32)]),
    (Resize2D25, [((1, 192, 40, 40), torch.float32)]),
    (Resize2D28, [((1, 384, 10, 10), torch.float32)]),
    (Resize2D24, [((1, 192, 20, 20), torch.float32)]),
    (Resize2D28, [((1, 256, 10, 10), torch.float32)]),
    (Resize2D28, [((1, 128, 10, 10), torch.float32)]),
    (Resize2D24, [((1, 64, 20, 20), torch.float32)]),
    (Resize2D30, [((1, 128, 13, 13), torch.float32)]),
    (Resize2D31, [((1, 64, 26, 26), torch.float32)]),
    (Resize2D30, [((1, 192, 13, 13), torch.float32)]),
    (Resize2D31, [((1, 96, 26, 26), torch.float32)]),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    forge_module, operand_shapes_dtypes = forge_module_and_shapes_dtypes

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(), dtype=parameter.pt_data_format
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(), dtype=constant.pt_data_format
        )
        framework_model.set_constant(name, constant_tensor)

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
