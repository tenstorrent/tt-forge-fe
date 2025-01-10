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


class Matmul0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, matmul_input_0, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, matmul_input_1)
        return matmul_output_1


class Matmul1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul1_const_1", shape=(1, 1, 6), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul1_const_1"))
        return matmul_output_1


class Matmul2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul2_const_1", shape=(1, 1, 334), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul2_const_1"))
        return matmul_output_1


class Matmul3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul3_const_1", shape=(1, 1, 7), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul3_const_1"))
        return matmul_output_1


class Matmul4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul4.weight_1",
            forge.Parameter(*(768, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul4.weight_1"))
        return matmul_output_1


class Matmul5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul5.weight_1",
            forge.Parameter(*(768, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul5.weight_1"))
        return matmul_output_1


class Matmul6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul6.weight_1",
            forge.Parameter(*(3072, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul6.weight_1"))
        return matmul_output_1


class Matmul7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul7_const_1", shape=(1, 1, 256), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul7_const_1"))
        return matmul_output_1


class Matmul8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul8_const_1", shape=(1, 1, 4), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul8_const_1"))
        return matmul_output_1


class Matmul9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul9_const_1", shape=(1, 1, 128), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul9_const_1"))
        return matmul_output_1


class Matmul10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul10_const_1", shape=(1, 1, 12), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul10_const_1"))
        return matmul_output_1


class Matmul11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul11_const_1", shape=(1, 1, 11), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul11_const_1"))
        return matmul_output_1


class Matmul12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul12_const_1", shape=(1, 1, 29), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul12_const_1"))
        return matmul_output_1


class Matmul13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul13_const_1", shape=(1, 1, 35), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul13_const_1"))
        return matmul_output_1


class Matmul14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul14_const_1", shape=(1, 1, 39), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul14_const_1"))
        return matmul_output_1


class Matmul15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul15_const_1", shape=(4, 24), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul15_const_1"))
        return matmul_output_1


class Matmul16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul16_const_1", shape=(4, 72), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul16_const_1"))
        return matmul_output_1


class Matmul17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul17_const_1", shape=(12, 24), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul17_const_1"))
        return matmul_output_1


class Matmul18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul18_const_1", shape=(12, 72), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul18_const_1"))
        return matmul_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Matmul0, [((2, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((32, 1, 64), torch.float32), ((32, 64, 1), torch.float32)]),
    (Matmul0, [((32, 1, 1), torch.float32), ((32, 1, 64), torch.float32)]),
    (Matmul0, [((26, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((24, 13, 64), torch.float32), ((24, 64, 13), torch.float32)]),
    (Matmul0, [((24, 13, 13), torch.float32), ((24, 13, 64), torch.float32)]),
    (Matmul0, [((26, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((26, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((26, 768), torch.float32), ((768, 1024), torch.float32)]),
    (Matmul0, [((26, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((32, 1, 64), torch.float32), ((32, 64, 13), torch.float32)]),
    (Matmul0, [((32, 1, 13), torch.float32), ((32, 13, 64), torch.float32)]),
    (Matmul0, [((2, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((2, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((2, 1024), torch.float32), ((1024, 2048), torch.float32)]),
    (Matmul0, [((2, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((64, 1, 64), torch.float32), ((64, 64, 1), torch.float32)]),
    (Matmul0, [((64, 1, 1), torch.float32), ((64, 1, 64), torch.float32)]),
    (Matmul0, [((26, 768), torch.float32), ((768, 2048), torch.float32)]),
    (Matmul0, [((26, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((64, 1, 64), torch.float32), ((64, 64, 13), torch.float32)]),
    (Matmul0, [((64, 1, 13), torch.float32), ((64, 13, 64), torch.float32)]),
    (Matmul0, [((2, 2048), torch.float32), ((2048, 8192), torch.float32)]),
    (Matmul0, [((2, 8192), torch.float32), ((8192, 2048), torch.float32)]),
    (Matmul0, [((2, 1536), torch.float32), ((1536, 1536), torch.float32)]),
    (Matmul0, [((48, 1, 64), torch.float32), ((48, 64, 1), torch.float32)]),
    (Matmul0, [((48, 1, 1), torch.float32), ((48, 1, 64), torch.float32)]),
    (Matmul0, [((26, 768), torch.float32), ((768, 1536), torch.float32)]),
    (Matmul0, [((26, 1536), torch.float32), ((1536, 1536), torch.float32)]),
    (Matmul0, [((48, 1, 64), torch.float32), ((48, 64, 13), torch.float32)]),
    (Matmul0, [((48, 1, 13), torch.float32), ((48, 13, 64), torch.float32)]),
    (Matmul0, [((2, 1536), torch.float32), ((1536, 6144), torch.float32)]),
    (Matmul0, [((2, 6144), torch.float32), ((6144, 1536), torch.float32)]),
    (Matmul0, [((2, 1536), torch.float32), ((1536, 2048), torch.float32)]),
    (Matmul0, [((16, 2, 64), torch.float32), ((16, 64, 2), torch.float32)]),
    (Matmul0, [((16, 2, 2), torch.float32), ((16, 2, 64), torch.float32)]),
    (Matmul0, [((1, 2, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((1500, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 2, 64), torch.float32), ((16, 64, 1500), torch.float32)]),
    (Matmul0, [((16, 2, 1500), torch.float32), ((16, 1500, 64), torch.float32)]),
    (Matmul0, [((1, 2, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 2, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((1, 2, 1024), torch.float32), ((1024, 51865), torch.float32)]),
    (Matmul0, [((2, 512), torch.float32), ((512, 512), torch.float32)]),
    (Matmul0, [((8, 2, 64), torch.float32), ((8, 64, 2), torch.float32)]),
    (Matmul0, [((8, 2, 2), torch.float32), ((8, 2, 64), torch.float32)]),
    (Matmul0, [((1, 2, 512), torch.float32), ((512, 512), torch.float32)]),
    (Matmul0, [((1500, 512), torch.float32), ((512, 512), torch.float32)]),
    (Matmul0, [((8, 2, 64), torch.float32), ((8, 64, 1500), torch.float32)]),
    (Matmul0, [((8, 2, 1500), torch.float32), ((8, 1500, 64), torch.float32)]),
    (Matmul0, [((1, 2, 512), torch.float32), ((512, 2048), torch.float32)]),
    (Matmul0, [((1, 2, 2048), torch.float32), ((2048, 512), torch.float32)]),
    (Matmul0, [((1, 2, 512), torch.float32), ((512, 51865), torch.float32)]),
    (Matmul0, [((2, 1280), torch.float32), ((1280, 1280), torch.float32)]),
    (Matmul0, [((20, 2, 64), torch.float32), ((20, 64, 2), torch.float32)]),
    (Matmul0, [((20, 2, 2), torch.float32), ((20, 2, 64), torch.float32)]),
    (Matmul0, [((1, 2, 1280), torch.float32), ((1280, 1280), torch.float32)]),
    (Matmul0, [((1500, 1280), torch.float32), ((1280, 1280), torch.float32)]),
    (Matmul0, [((20, 2, 64), torch.float32), ((20, 64, 1500), torch.float32)]),
    (Matmul0, [((20, 2, 1500), torch.float32), ((20, 1500, 64), torch.float32)]),
    (Matmul0, [((1, 2, 1280), torch.float32), ((1280, 5120), torch.float32)]),
    (Matmul0, [((1, 2, 5120), torch.float32), ((5120, 1280), torch.float32)]),
    (Matmul0, [((1, 2, 1280), torch.float32), ((1280, 51865), torch.float32)]),
    (Matmul0, [((2, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 2, 64), torch.float32), ((12, 64, 2), torch.float32)]),
    (Matmul0, [((12, 2, 2), torch.float32), ((12, 2, 64), torch.float32)]),
    (Matmul0, [((1, 2, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((1500, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 2, 64), torch.float32), ((12, 64, 1500), torch.float32)]),
    (Matmul0, [((12, 2, 1500), torch.float32), ((12, 1500, 64), torch.float32)]),
    (Matmul0, [((1, 2, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 2, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 2, 768), torch.float32), ((768, 51865), torch.float32)]),
    (Matmul0, [((2, 384), torch.float32), ((384, 384), torch.float32)]),
    (Matmul0, [((6, 2, 64), torch.float32), ((6, 64, 2), torch.float32)]),
    (Matmul0, [((6, 2, 2), torch.float32), ((6, 2, 64), torch.float32)]),
    (Matmul0, [((1, 2, 384), torch.float32), ((384, 384), torch.float32)]),
    (Matmul0, [((1500, 384), torch.float32), ((384, 384), torch.float32)]),
    (Matmul0, [((6, 2, 64), torch.float32), ((6, 64, 1500), torch.float32)]),
    (Matmul0, [((6, 2, 1500), torch.float32), ((6, 1500, 64), torch.float32)]),
    (Matmul0, [((1, 2, 384), torch.float32), ((384, 1536), torch.float32)]),
    (Matmul0, [((1, 2, 1536), torch.float32), ((1536, 384), torch.float32)]),
    (Matmul0, [((1, 2, 384), torch.float32), ((384, 51865), torch.float32)]),
    (Matmul0, [((1, 2, 1280), torch.float32), ((1280, 51866), torch.float32)]),
    (Matmul0, [((14, 512), torch.float32), ((512, 512), torch.float32)]),
    (Matmul0, [((16, 7, 64), torch.float32), ((16, 64, 7), torch.float32)]),
    (Matmul0, [((16, 7, 7), torch.float32), ((16, 7, 64), torch.float32)]),
    (Matmul0, [((14, 512), torch.float32), ((512, 2048), torch.float32)]),
    (Matmul0, [((14, 2048), torch.float32), ((2048, 512), torch.float32)]),
    (Matmul0, [((204, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 204, 64), torch.float32), ((12, 64, 204), torch.float32)]),
    (Matmul0, [((12, 204, 204), torch.float32), ((12, 204, 64), torch.float32)]),
    (Matmul0, [((1, 204, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 204, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 11, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((1, 11, 768), torch.float32), ((768, 30522), torch.float32)]),
    (Matmul0, [((201, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 201, 64), torch.float32), ((12, 64, 201), torch.float32)]),
    (Matmul0, [((12, 201, 201), torch.float32), ((12, 201, 64), torch.float32)]),
    (Matmul0, [((1, 201, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 201, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 1536), torch.float32)]),
    (Matmul0, [((1, 1536), torch.float32), ((1536, 3129), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 2048), torch.float32)]),
    (Matmul0, [((128, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((16, 128, 128), torch.float32), ((16, 128, 128), torch.float32)]),
    (Matmul0, [((1, 128, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((1, 128, 2048), torch.float32), ((2048, 8192), torch.float32)]),
    (Matmul0, [((1, 128, 8192), torch.float32), ((8192, 2048), torch.float32)]),
    (Matmul0, [((1, 128, 2048), torch.float32), ((2048, 2), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 1024), torch.float32)]),
    (Matmul0, [((128, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 128, 64), torch.float32), ((16, 64, 128), torch.float32)]),
    (Matmul0, [((16, 128, 128), torch.float32), ((16, 128, 64), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 2), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 4096), torch.float32)]),
    (Matmul0, [((128, 4096), torch.float32), ((4096, 4096), torch.float32)]),
    (Matmul0, [((64, 128, 64), torch.float32), ((64, 64, 128), torch.float32)]),
    (Matmul0, [((64, 128, 128), torch.float32), ((64, 128, 64), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 4096), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 16384), torch.float32)]),
    (Matmul0, [((1, 128, 16384), torch.float32), ((16384, 4096), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 128), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 30000), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 2), torch.float32)]),
    (Matmul0, [((1, 128, 2048), torch.float32), ((2048, 128), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 128), torch.float32)]),
    (Matmul0, [((1, 128, 128), torch.float32), ((128, 768), torch.float32)]),
    (Matmul0, [((128, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 128, 64), torch.float32), ((12, 64, 128), torch.float32)]),
    (Matmul0, [((12, 128, 128), torch.float32), ((12, 128, 64), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 128, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 2), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 128), torch.float32)]),
    (Matmul0, [((256, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 256, 64), torch.float32), ((16, 64, 256), torch.float32)]),
    (Matmul0, [((16, 256, 256), torch.float32), ((16, 256, 64), torch.float32)]),
    (Matmul0, [((1, 256, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((1, 256, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 256, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((384, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 384, 64), torch.float32), ((16, 64, 384), torch.float32)]),
    (Matmul0, [((16, 384, 384), torch.float32), ((16, 384, 64), torch.float32)]),
    (Matmul0, [((1, 384, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 384, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((384, 1024), torch.float32), ((1024, 1), torch.float32)]),
    (Matmul0, [((1, 128, 1024), torch.float32), ((1024, 9), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 30522), torch.float32)]),
    (Matmul0, [((256, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 256, 1024), torch.float32), ((1024, 51200), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 2), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 119547), torch.float32)]),
    (Matmul0, [((384, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 384, 64), torch.float32), ((12, 64, 384), torch.float32)]),
    (Matmul0, [((12, 384, 384), torch.float32), ((12, 384, 64), torch.float32)]),
    (Matmul0, [((1, 384, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 384, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((384, 768), torch.float32), ((768, 1), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 9), torch.float32)]),
    (Matmul0, [((128, 768), torch.float32), ((768, 1), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 1), torch.float32)]),
    (Matmul0, [((6, 4544), torch.float32), ((4544, 18176), torch.float32)]),
    (Matmul0, [((1, 6, 18176), torch.float32), ((18176, 4544), torch.float32)]),
    (Matmul0, [((6, 4544), torch.float32), ((4544, 4672), torch.float32)]),
    (Matmul1, [((1, 32, 1), torch.float32)]),
    (Matmul0, [((71, 6, 64), torch.float32), ((1, 64, 6), torch.float32)]),
    (Matmul0, [((71, 6, 6), torch.float32), ((1, 6, 64), torch.float32)]),
    (Matmul0, [((6, 4544), torch.float32), ((4544, 4544), torch.float32)]),
    (Matmul0, [((1, 6, 4544), torch.float32), ((4544, 65024), torch.float32)]),
    (Matmul0, [((1, 334, 4096), torch.float32), ((4096, 12288), torch.float32)]),
    (Matmul2, [((1, 16, 1), torch.float32)]),
    (Matmul0, [((64, 334, 64), torch.float32), ((64, 64, 334), torch.float32)]),
    (Matmul0, [((64, 334, 334), torch.float32), ((64, 334, 64), torch.float32)]),
    (Matmul0, [((334, 4096), torch.float32), ((4096, 4096), torch.float32)]),
    (Matmul0, [((1, 334, 4096), torch.float32), ((4096, 16384), torch.float32)]),
    (Matmul0, [((1, 334, 16384), torch.float32), ((16384, 4096), torch.float32)]),
    (Matmul0, [((7, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul3, [((1, 128, 1), torch.float32)]),
    (Matmul0, [((7, 2048), torch.float32), ((2048, 256), torch.float32)]),
    (Matmul0, [((8, 7, 256), torch.float32), ((8, 256, 7), torch.float32)]),
    (Matmul0, [((8, 7, 7), torch.float32), ((8, 7, 256), torch.float32)]),
    (Matmul0, [((7, 2048), torch.float32), ((2048, 16384), torch.float32)]),
    (Matmul0, [((1, 7, 16384), torch.float32), ((16384, 2048), torch.float32)]),
    (Matmul0, [((1, 7, 2048), torch.float32), ((2048, 256000), torch.float32)]),
    (Matmul0, [((256, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 256, 64), torch.float32), ((12, 64, 256), torch.float32)]),
    (Matmul0, [((12, 256, 256), torch.float32), ((12, 256, 64), torch.float32)]),
    (Matmul4, [((256, 768), torch.float32)]),
    (Matmul5, [((256, 768), torch.float32)]),
    (Matmul6, [((256, 3072), torch.float32)]),
    (Matmul0, [((1, 256, 768), torch.float32), ((768, 50257), torch.float32)]),
    (Matmul0, [((32, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 32, 64), torch.float32), ((12, 64, 32), torch.float32)]),
    (Matmul0, [((12, 32, 32), torch.float32), ((12, 32, 64), torch.float32)]),
    (Matmul0, [((1, 32, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 32, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 32, 768), torch.float32), ((768, 2), torch.float32)]),
    (Matmul0, [((32, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((16, 32, 128), torch.float32), ((16, 128, 32), torch.float32)]),
    (Matmul0, [((16, 32, 32), torch.float32), ((16, 32, 128), torch.float32)]),
    (Matmul0, [((1, 32, 2048), torch.float32), ((2048, 8192), torch.float32)]),
    (Matmul0, [((1, 32, 8192), torch.float32), ((8192, 2048), torch.float32)]),
    (Matmul0, [((1, 32, 2048), torch.float32), ((2048, 2), torch.float32)]),
    (Matmul0, [((1, 256, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 256, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((256, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((16, 256, 128), torch.float32), ((16, 128, 256), torch.float32)]),
    (Matmul0, [((16, 256, 256), torch.float32), ((16, 256, 128), torch.float32)]),
    (Matmul0, [((1, 256, 2048), torch.float32), ((2048, 8192), torch.float32)]),
    (Matmul0, [((1, 256, 8192), torch.float32), ((8192, 2048), torch.float32)]),
    (Matmul0, [((1, 256, 2048), torch.float32), ((2048, 50257), torch.float32)]),
    (Matmul0, [((256, 2560), torch.float32), ((2560, 2560), torch.float32)]),
    (Matmul0, [((20, 256, 128), torch.float32), ((20, 128, 256), torch.float32)]),
    (Matmul0, [((20, 256, 256), torch.float32), ((20, 256, 128), torch.float32)]),
    (Matmul0, [((1, 256, 2560), torch.float32), ((2560, 10240), torch.float32)]),
    (Matmul0, [((1, 256, 10240), torch.float32), ((10240, 2560), torch.float32)]),
    (Matmul0, [((1, 256, 2560), torch.float32), ((2560, 50257), torch.float32)]),
    (Matmul0, [((32, 2560), torch.float32), ((2560, 2560), torch.float32)]),
    (Matmul0, [((20, 32, 128), torch.float32), ((20, 128, 32), torch.float32)]),
    (Matmul0, [((20, 32, 32), torch.float32), ((20, 32, 128), torch.float32)]),
    (Matmul0, [((1, 32, 2560), torch.float32), ((2560, 10240), torch.float32)]),
    (Matmul0, [((1, 32, 10240), torch.float32), ((10240, 2560), torch.float32)]),
    (Matmul0, [((1, 32, 2560), torch.float32), ((2560, 2), torch.float32)]),
    (Matmul0, [((256, 4096), torch.float32), ((4096, 4096), torch.float32)]),
    (Matmul7, [((1, 64, 1), torch.float32)]),
    (Matmul0, [((256, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((32, 256, 128), torch.float32), ((32, 128, 256), torch.float32)]),
    (Matmul0, [((32, 256, 256), torch.float32), ((32, 256, 128), torch.float32)]),
    (Matmul0, [((256, 4096), torch.float32), ((4096, 14336), torch.float32)]),
    (Matmul0, [((1, 256, 14336), torch.float32), ((14336, 4096), torch.float32)]),
    (Matmul0, [((1, 256, 4096), torch.float32), ((4096, 128256), torch.float32)]),
    (Matmul0, [((4, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul8, [((1, 32, 1), torch.float32)]),
    (Matmul0, [((4, 2048), torch.float32), ((2048, 512), torch.float32)]),
    (Matmul0, [((32, 4, 64), torch.float32), ((32, 64, 4), torch.float32)]),
    (Matmul0, [((32, 4, 4), torch.float32), ((32, 4, 64), torch.float32)]),
    (Matmul0, [((4, 2048), torch.float32), ((2048, 8192), torch.float32)]),
    (Matmul0, [((1, 4, 8192), torch.float32), ((8192, 2048), torch.float32)]),
    (Matmul0, [((1, 4, 2048), torch.float32), ((2048, 2), torch.float32)]),
    (Matmul0, [((4, 4096), torch.float32), ((4096, 4096), torch.float32)]),
    (Matmul8, [((1, 64, 1), torch.float32)]),
    (Matmul0, [((4, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((32, 4, 128), torch.float32), ((32, 128, 4), torch.float32)]),
    (Matmul0, [((32, 4, 4), torch.float32), ((32, 4, 128), torch.float32)]),
    (Matmul0, [((4, 4096), torch.float32), ((4096, 14336), torch.float32)]),
    (Matmul0, [((1, 4, 14336), torch.float32), ((14336, 4096), torch.float32)]),
    (Matmul0, [((1, 4, 4096), torch.float32), ((4096, 2), torch.float32)]),
    (Matmul7, [((1, 32, 1), torch.float32)]),
    (Matmul0, [((256, 2048), torch.float32), ((2048, 512), torch.float32)]),
    (Matmul0, [((32, 256, 64), torch.float32), ((32, 64, 256), torch.float32)]),
    (Matmul0, [((32, 256, 256), torch.float32), ((32, 256, 64), torch.float32)]),
    (Matmul0, [((256, 2048), torch.float32), ((2048, 8192), torch.float32)]),
    (Matmul0, [((1, 256, 2048), torch.float32), ((2048, 128256), torch.float32)]),
    (Matmul9, [((1, 64, 1), torch.float32)]),
    (Matmul0, [((128, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((32, 128, 128), torch.float32), ((32, 128, 128), torch.float32)]),
    (Matmul0, [((128, 4096), torch.float32), ((4096, 14336), torch.float32)]),
    (Matmul0, [((1, 128, 14336), torch.float32), ((14336, 4096), torch.float32)]),
    (Matmul0, [((1, 128, 4096), torch.float32), ((4096, 32000), torch.float32)]),
    (Matmul0, [((32, 32, 64), torch.float32), ((32, 64, 32), torch.float32)]),
    (Matmul0, [((32, 32, 32), torch.float32), ((32, 32, 64), torch.float32)]),
    (Matmul0, [((32, 2048), torch.float32), ((2048, 8192), torch.float32)]),
    (Matmul0, [((32, 8192), torch.float32), ((8192, 2048), torch.float32)]),
    (Matmul0, [((32, 2048), torch.float32), ((2048, 1), torch.float32)]),
    (Matmul0, [((256, 8192), torch.float32), ((8192, 2048), torch.float32)]),
    (Matmul0, [((1, 256, 2048), torch.float32), ((2048, 50272), torch.float32)]),
    (Matmul0, [((1, 32, 512), torch.float32), ((512, 1024), torch.float32)]),
    (Matmul0, [((32, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 32, 64), torch.float32), ((16, 64, 32), torch.float32)]),
    (Matmul0, [((16, 32, 32), torch.float32), ((16, 32, 64), torch.float32)]),
    (Matmul0, [((32, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((32, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((32, 1024), torch.float32), ((1024, 512), torch.float32)]),
    (Matmul0, [((32, 512), torch.float32), ((512, 1), torch.float32)]),
    (Matmul0, [((32, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((32, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((32, 768), torch.float32), ((768, 2), torch.float32)]),
    (Matmul0, [((32, 2048), torch.float32), ((2048, 2), torch.float32)]),
    (Matmul0, [((256, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((256, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 256, 768), torch.float32), ((768, 50272), torch.float32)]),
    (Matmul0, [((32, 512), torch.float32), ((512, 2), torch.float32)]),
    (Matmul0, [((32, 768), torch.float32), ((768, 1), torch.float32)]),
    (Matmul0, [((1, 256, 512), torch.float32), ((512, 1024), torch.float32)]),
    (Matmul0, [((256, 1024), torch.float32), ((1024, 512), torch.float32)]),
    (Matmul0, [((256, 512), torch.float32), ((512, 50272), torch.float32)]),
    (Matmul7, [((1, 16, 1), torch.float32)]),
    (Matmul0, [((32, 256, 80), torch.float32), ((32, 80, 256), torch.float32)]),
    (Matmul0, [((32, 256, 256), torch.float32), ((32, 256, 80), torch.float32)]),
    (Matmul0, [((256, 2560), torch.float32), ((2560, 10240), torch.float32)]),
    (Matmul0, [((1, 256, 2560), torch.float32), ((2560, 51200), torch.float32)]),
    (Matmul0, [((12, 2560), torch.float32), ((2560, 2560), torch.float32)]),
    (Matmul10, [((1, 16, 1), torch.float32)]),
    (Matmul0, [((32, 12, 80), torch.float32), ((32, 80, 12), torch.float32)]),
    (Matmul0, [((32, 12, 12), torch.float32), ((32, 12, 80), torch.float32)]),
    (Matmul0, [((12, 2560), torch.float32), ((2560, 10240), torch.float32)]),
    (Matmul0, [((1, 12, 10240), torch.float32), ((10240, 2560), torch.float32)]),
    (Matmul0, [((1, 12, 2560), torch.float32), ((2560, 2), torch.float32)]),
    (Matmul0, [((11, 2560), torch.float32), ((2560, 2560), torch.float32)]),
    (Matmul11, [((1, 16, 1), torch.float32)]),
    (Matmul0, [((32, 11, 80), torch.float32), ((32, 80, 11), torch.float32)]),
    (Matmul0, [((32, 11, 11), torch.float32), ((32, 11, 80), torch.float32)]),
    (Matmul0, [((11, 2560), torch.float32), ((2560, 10240), torch.float32)]),
    (Matmul0, [((1, 11, 10240), torch.float32), ((10240, 2560), torch.float32)]),
    (Matmul0, [((1, 11, 2560), torch.float32), ((2560, 2), torch.float32)]),
    (Matmul0, [((6, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 6, 64), torch.float32), ((16, 64, 6), torch.float32)]),
    (Matmul0, [((16, 6, 6), torch.float32), ((16, 6, 64), torch.float32)]),
    (Matmul0, [((6, 1024), torch.float32), ((1024, 2816), torch.float32)]),
    (Matmul0, [((1, 6, 2816), torch.float32), ((2816, 1024), torch.float32)]),
    (Matmul0, [((1, 6, 1024), torch.float32), ((1024, 151936), torch.float32)]),
    (Matmul0, [((29, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul12, [((1, 32, 1), torch.float32)]),
    (Matmul0, [((16, 29, 64), torch.float32), ((16, 64, 29), torch.float32)]),
    (Matmul0, [((16, 29, 29), torch.float32), ((16, 29, 64), torch.float32)]),
    (Matmul0, [((29, 1024), torch.float32), ((1024, 2816), torch.float32)]),
    (Matmul0, [((1, 29, 2816), torch.float32), ((2816, 1024), torch.float32)]),
    (Matmul0, [((1, 29, 1024), torch.float32), ((1024, 151936), torch.float32)]),
    (Matmul0, [((35, 1536), torch.float32), ((1536, 1536), torch.float32)]),
    (Matmul13, [((1, 64, 1), torch.float32)]),
    (Matmul0, [((35, 1536), torch.float32), ((1536, 256), torch.float32)]),
    (Matmul0, [((12, 35, 128), torch.float32), ((12, 128, 35), torch.float32)]),
    (Matmul0, [((12, 35, 35), torch.float32), ((12, 35, 128), torch.float32)]),
    (Matmul0, [((35, 1536), torch.float32), ((1536, 8960), torch.float32)]),
    (Matmul0, [((1, 35, 8960), torch.float32), ((8960, 1536), torch.float32)]),
    (Matmul0, [((1, 35, 1536), torch.float32), ((1536, 151936), torch.float32)]),
    (Matmul0, [((35, 3584), torch.float32), ((3584, 3584), torch.float32)]),
    (Matmul0, [((35, 3584), torch.float32), ((3584, 512), torch.float32)]),
    (Matmul0, [((28, 35, 128), torch.float32), ((28, 128, 35), torch.float32)]),
    (Matmul0, [((28, 35, 35), torch.float32), ((28, 35, 128), torch.float32)]),
    (Matmul0, [((35, 3584), torch.float32), ((3584, 18944), torch.float32)]),
    (Matmul0, [((1, 35, 18944), torch.float32), ((18944, 3584), torch.float32)]),
    (Matmul0, [((1, 35, 3584), torch.float32), ((3584, 152064), torch.float32)]),
    (Matmul0, [((35, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((35, 2048), torch.float32), ((2048, 256), torch.float32)]),
    (Matmul0, [((16, 35, 128), torch.float32), ((16, 128, 35), torch.float32)]),
    (Matmul0, [((16, 35, 35), torch.float32), ((16, 35, 128), torch.float32)]),
    (Matmul0, [((35, 2048), torch.float32), ((2048, 11008), torch.float32)]),
    (Matmul0, [((1, 35, 11008), torch.float32), ((11008, 2048), torch.float32)]),
    (Matmul0, [((1, 35, 2048), torch.float32), ((2048, 151936), torch.float32)]),
    (Matmul0, [((35, 896), torch.float32), ((896, 896), torch.float32)]),
    (Matmul13, [((1, 32, 1), torch.float32)]),
    (Matmul0, [((35, 896), torch.float32), ((896, 128), torch.float32)]),
    (Matmul0, [((14, 35, 64), torch.float32), ((14, 64, 35), torch.float32)]),
    (Matmul0, [((14, 35, 35), torch.float32), ((14, 35, 64), torch.float32)]),
    (Matmul0, [((35, 896), torch.float32), ((896, 4864), torch.float32)]),
    (Matmul0, [((1, 35, 4864), torch.float32), ((4864, 896), torch.float32)]),
    (Matmul0, [((1, 35, 896), torch.float32), ((896, 151936), torch.float32)]),
    (Matmul0, [((29, 896), torch.float32), ((896, 896), torch.float32)]),
    (Matmul0, [((29, 896), torch.float32), ((896, 128), torch.float32)]),
    (Matmul0, [((14, 29, 64), torch.float32), ((14, 64, 29), torch.float32)]),
    (Matmul0, [((14, 29, 29), torch.float32), ((14, 29, 64), torch.float32)]),
    (Matmul0, [((29, 896), torch.float32), ((896, 4864), torch.float32)]),
    (Matmul0, [((1, 29, 4864), torch.float32), ((4864, 896), torch.float32)]),
    (Matmul0, [((1, 29, 896), torch.float32), ((896, 151936), torch.float32)]),
    (Matmul0, [((39, 1536), torch.float32), ((1536, 1536), torch.float32)]),
    (Matmul14, [((1, 64, 1), torch.float32)]),
    (Matmul0, [((39, 1536), torch.float32), ((1536, 256), torch.float32)]),
    (Matmul0, [((12, 39, 128), torch.float32), ((12, 128, 39), torch.float32)]),
    (Matmul0, [((12, 39, 39), torch.float32), ((12, 39, 128), torch.float32)]),
    (Matmul0, [((39, 1536), torch.float32), ((1536, 8960), torch.float32)]),
    (Matmul0, [((1, 39, 8960), torch.float32), ((8960, 1536), torch.float32)]),
    (Matmul0, [((1, 39, 1536), torch.float32), ((1536, 151936), torch.float32)]),
    (Matmul0, [((29, 1536), torch.float32), ((1536, 1536), torch.float32)]),
    (Matmul12, [((1, 64, 1), torch.float32)]),
    (Matmul0, [((29, 1536), torch.float32), ((1536, 256), torch.float32)]),
    (Matmul0, [((12, 29, 128), torch.float32), ((12, 128, 29), torch.float32)]),
    (Matmul0, [((12, 29, 29), torch.float32), ((12, 29, 128), torch.float32)]),
    (Matmul0, [((29, 1536), torch.float32), ((1536, 8960), torch.float32)]),
    (Matmul0, [((1, 29, 8960), torch.float32), ((8960, 1536), torch.float32)]),
    (Matmul0, [((1, 29, 1536), torch.float32), ((1536, 151936), torch.float32)]),
    (Matmul0, [((29, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((29, 2048), torch.float32), ((2048, 256), torch.float32)]),
    (Matmul0, [((16, 29, 128), torch.float32), ((16, 128, 29), torch.float32)]),
    (Matmul0, [((16, 29, 29), torch.float32), ((16, 29, 128), torch.float32)]),
    (Matmul0, [((29, 2048), torch.float32), ((2048, 11008), torch.float32)]),
    (Matmul0, [((1, 29, 11008), torch.float32), ((11008, 2048), torch.float32)]),
    (Matmul0, [((1, 29, 2048), torch.float32), ((2048, 151936), torch.float32)]),
    (Matmul0, [((39, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((39, 2048), torch.float32), ((2048, 256), torch.float32)]),
    (Matmul0, [((16, 39, 128), torch.float32), ((16, 128, 39), torch.float32)]),
    (Matmul0, [((16, 39, 39), torch.float32), ((16, 39, 128), torch.float32)]),
    (Matmul0, [((39, 2048), torch.float32), ((2048, 11008), torch.float32)]),
    (Matmul0, [((1, 39, 11008), torch.float32), ((11008, 2048), torch.float32)]),
    (Matmul0, [((1, 39, 2048), torch.float32), ((2048, 151936), torch.float32)]),
    (Matmul0, [((39, 896), torch.float32), ((896, 896), torch.float32)]),
    (Matmul14, [((1, 32, 1), torch.float32)]),
    (Matmul0, [((39, 896), torch.float32), ((896, 128), torch.float32)]),
    (Matmul0, [((14, 39, 64), torch.float32), ((14, 64, 39), torch.float32)]),
    (Matmul0, [((14, 39, 39), torch.float32), ((14, 39, 64), torch.float32)]),
    (Matmul0, [((39, 896), torch.float32), ((896, 4864), torch.float32)]),
    (Matmul0, [((1, 39, 4864), torch.float32), ((4864, 896), torch.float32)]),
    (Matmul0, [((1, 39, 896), torch.float32), ((896, 151936), torch.float32)]),
    (Matmul0, [((39, 3584), torch.float32), ((3584, 3584), torch.float32)]),
    (Matmul0, [((39, 3584), torch.float32), ((3584, 512), torch.float32)]),
    (Matmul0, [((28, 39, 128), torch.float32), ((28, 128, 39), torch.float32)]),
    (Matmul0, [((28, 39, 39), torch.float32), ((28, 39, 128), torch.float32)]),
    (Matmul0, [((39, 3584), torch.float32), ((3584, 18944), torch.float32)]),
    (Matmul0, [((1, 39, 18944), torch.float32), ((18944, 3584), torch.float32)]),
    (Matmul0, [((1, 39, 3584), torch.float32), ((3584, 152064), torch.float32)]),
    (Matmul0, [((29, 3584), torch.float32), ((3584, 3584), torch.float32)]),
    (Matmul0, [((29, 3584), torch.float32), ((3584, 512), torch.float32)]),
    (Matmul0, [((28, 29, 128), torch.float32), ((28, 128, 29), torch.float32)]),
    (Matmul0, [((28, 29, 29), torch.float32), ((28, 29, 128), torch.float32)]),
    (Matmul0, [((29, 3584), torch.float32), ((3584, 18944), torch.float32)]),
    (Matmul0, [((1, 29, 18944), torch.float32), ((18944, 3584), torch.float32)]),
    (Matmul0, [((1, 29, 3584), torch.float32), ((3584, 152064), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 3), torch.float32)]),
    (Matmul0, [((1, 128, 768), torch.float32), ((768, 250002), torch.float32)]),
    (Matmul0, [((1, 512), torch.float32), ((512, 384), torch.float32)]),
    (Matmul0, [((6, 1, 64), torch.float32), ((6, 64, 1), torch.float32)]),
    (Matmul0, [((6, 1, 1), torch.float32), ((6, 1, 64), torch.float32)]),
    (Matmul0, [((1, 384), torch.float32), ((384, 512), torch.float32)]),
    (Matmul0, [((1, 512), torch.float32), ((512, 1024), torch.float32)]),
    (Matmul0, [((1, 1, 1024), torch.float32), ((1024, 512), torch.float32)]),
    (Matmul0, [((1, 1, 512), torch.float32), ((512, 32128), torch.float32)]),
    (Matmul0, [((12, 1, 64), torch.float32), ((12, 64, 1), torch.float32)]),
    (Matmul0, [((12, 1, 1), torch.float32), ((12, 1, 64), torch.float32)]),
    (Matmul0, [((12, 1, 64), torch.float32), ((12, 64, 256), torch.float32)]),
    (Matmul0, [((12, 1, 256), torch.float32), ((12, 256, 64), torch.float32)]),
    (Matmul0, [((1, 1, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 1, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 1, 768), torch.float32), ((768, 32128), torch.float32)]),
    (Matmul0, [((1, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 1, 64), torch.float32), ((16, 64, 1), torch.float32)]),
    (Matmul0, [((16, 1, 1), torch.float32), ((16, 1, 64), torch.float32)]),
    (Matmul0, [((16, 1, 64), torch.float32), ((16, 64, 256), torch.float32)]),
    (Matmul0, [((16, 1, 256), torch.float32), ((16, 256, 64), torch.float32)]),
    (Matmul0, [((1, 1, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 1, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((1, 1, 1024), torch.float32), ((1024, 32128), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 2048), torch.float32)]),
    (Matmul0, [((1, 1, 2048), torch.float32), ((2048, 768), torch.float32)]),
    (Matmul0, [((1, 512), torch.float32), ((512, 512), torch.float32)]),
    (Matmul0, [((8, 1, 64), torch.float32), ((8, 64, 1), torch.float32)]),
    (Matmul0, [((8, 1, 1), torch.float32), ((8, 1, 64), torch.float32)]),
    (Matmul0, [((1, 1, 512), torch.float32), ((512, 2048), torch.float32)]),
    (Matmul0, [((1, 1, 2048), torch.float32), ((2048, 512), torch.float32)]),
    (Matmul0, [((1, 256, 2048), torch.float32), ((2048, 256008), torch.float32)]),
    (Matmul0, [((1, 256, 1024), torch.float32), ((1024, 256008), torch.float32)]),
    (Matmul0, [((1024, 72), torch.float32), ((72, 512), torch.float32)]),
    (Matmul0, [((1024, 512), torch.float32), ((512, 512), torch.float32)]),
    (Matmul0, [((1024, 512), torch.float32), ((512, 96), torch.float32)]),
    (Matmul0, [((1024, 72), torch.float32), ((72, 256), torch.float32)]),
    (Matmul0, [((1024, 256), torch.float32), ((256, 256), torch.float32)]),
    (Matmul0, [((1024, 256), torch.float32), ((256, 8), torch.float32)]),
    (Matmul15, [((1024, 4), torch.float32)]),
    (Matmul16, [((1024, 4), torch.float32)]),
    (Matmul0, [((1024, 72), torch.float32), ((72, 2048), torch.float32)]),
    (Matmul0, [((1024, 2048), torch.float32), ((2048, 2048), torch.float32)]),
    (Matmul0, [((1024, 2048), torch.float32), ((2048, 48), torch.float32)]),
    (Matmul17, [((1024, 12), torch.float32)]),
    (Matmul18, [((1024, 12), torch.float32)]),
    (Matmul0, [((1, 9216), torch.float32), ((9216, 4096), torch.float32)]),
    (Matmul0, [((1, 4096), torch.float32), ((4096, 4096), torch.float32)]),
    (Matmul0, [((1, 4096), torch.float32), ((4096, 1000), torch.float32)]),
    (Matmul0, [((1, 784), torch.float32), ((784, 128), torch.float32)]),
    (Matmul0, [((1, 128), torch.float32), ((128, 64), torch.float32)]),
    (Matmul0, [((1, 64), torch.float32), ((64, 12), torch.float32)]),
    (Matmul0, [((1, 12), torch.float32), ((12, 3), torch.float32)]),
    (Matmul0, [((1, 3), torch.float32), ((3, 12), torch.float32)]),
    (Matmul0, [((1, 12), torch.float32), ((12, 64), torch.float32)]),
    (Matmul0, [((1, 64), torch.float32), ((64, 128), torch.float32)]),
    (Matmul0, [((1, 128), torch.float32), ((128, 784), torch.float32)]),
    (Matmul0, [((197, 768), torch.float32), ((768, 768), torch.float32)]),
    (Matmul0, [((12, 197, 64), torch.float32), ((12, 64, 197), torch.float32)]),
    (Matmul0, [((12, 197, 197), torch.float32), ((12, 197, 64), torch.float32)]),
    (Matmul0, [((1, 197, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 197, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 1000), torch.float32)]),
    (Matmul0, [((197, 192), torch.float32), ((192, 192), torch.float32)]),
    (Matmul0, [((3, 197, 64), torch.float32), ((3, 64, 197), torch.float32)]),
    (Matmul0, [((3, 197, 197), torch.float32), ((3, 197, 64), torch.float32)]),
    (Matmul0, [((1, 197, 192), torch.float32), ((192, 768), torch.float32)]),
    (Matmul0, [((1, 197, 768), torch.float32), ((768, 192), torch.float32)]),
    (Matmul0, [((1, 192), torch.float32), ((192, 1000), torch.float32)]),
    (Matmul0, [((197, 384), torch.float32), ((384, 384), torch.float32)]),
    (Matmul0, [((6, 197, 64), torch.float32), ((6, 64, 197), torch.float32)]),
    (Matmul0, [((6, 197, 197), torch.float32), ((6, 197, 64), torch.float32)]),
    (Matmul0, [((1, 197, 384), torch.float32), ((384, 1536), torch.float32)]),
    (Matmul0, [((1, 197, 1536), torch.float32), ((1536, 384), torch.float32)]),
    (Matmul0, [((1, 384), torch.float32), ((384, 1000), torch.float32)]),
    (Matmul0, [((1, 1920), torch.float32), ((1920, 1000), torch.float32)]),
    (Matmul0, [((1, 1024), torch.float32), ((1024, 1000), torch.float32)]),
    (Matmul0, [((1, 2208), torch.float32), ((2208, 1000), torch.float32)]),
    (Matmul0, [((1, 1664), torch.float32), ((1664, 1000), torch.float32)]),
    (Matmul0, [((1, 1792), torch.float32), ((1792, 1000), torch.float32)]),
    (Matmul0, [((1, 1280), torch.float32), ((1280, 1000), torch.float32)]),
    (Matmul0, [((1, 2048), torch.float32), ((2048, 1000), torch.float32)]),
    (Matmul0, [((1, 1536), torch.float32), ((1536, 1000), torch.float32)]),
    (Matmul0, [((1, 768, 49), torch.float32), ((49, 384), torch.float32)]),
    (Matmul0, [((1, 768, 384), torch.float32), ((384, 49), torch.float32)]),
    (Matmul0, [((1, 49, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 49, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 1024, 49), torch.float32), ((49, 512), torch.float32)]),
    (Matmul0, [((1, 1024, 512), torch.float32), ((512, 49), torch.float32)]),
    (Matmul0, [((1, 49, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 49, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((1, 768, 196), torch.float32), ((196, 384), torch.float32)]),
    (Matmul0, [((1, 768, 384), torch.float32), ((384, 196), torch.float32)]),
    (Matmul0, [((1, 196, 768), torch.float32), ((768, 3072), torch.float32)]),
    (Matmul0, [((1, 196, 3072), torch.float32), ((3072, 768), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 11221), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 21843), torch.float32)]),
    (Matmul0, [((1, 512, 49), torch.float32), ((49, 256), torch.float32)]),
    (Matmul0, [((1, 512, 256), torch.float32), ((256, 49), torch.float32)]),
    (Matmul0, [((1, 49, 512), torch.float32), ((512, 2048), torch.float32)]),
    (Matmul0, [((1, 49, 2048), torch.float32), ((2048, 512), torch.float32)]),
    (Matmul0, [((1, 512), torch.float32), ((512, 1000), torch.float32)]),
    (Matmul0, [((1, 512, 196), torch.float32), ((196, 256), torch.float32)]),
    (Matmul0, [((1, 512, 256), torch.float32), ((256, 196), torch.float32)]),
    (Matmul0, [((1, 196, 512), torch.float32), ((512, 2048), torch.float32)]),
    (Matmul0, [((1, 196, 2048), torch.float32), ((2048, 512), torch.float32)]),
    (Matmul0, [((1, 1024, 196), torch.float32), ((196, 512), torch.float32)]),
    (Matmul0, [((1, 1024, 512), torch.float32), ((512, 196), torch.float32)]),
    (Matmul0, [((1, 196, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 196, 4096), torch.float32), ((4096, 1024), torch.float32)]),
    (Matmul0, [((1, 1024), torch.float32), ((1024, 21843), torch.float32)]),
    (Matmul0, [((1, 768), torch.float32), ((768, 1001), torch.float32)]),
    (Matmul0, [((1, 1024), torch.float32), ((1024, 9), torch.float32)]),
    (Matmul0, [((1, 1024), torch.float32), ((1024, 1001), torch.float32)]),
    (Matmul0, [((1, 1280), torch.float32), ((1280, 1001), torch.float32)]),
    (Matmul0, [((1, 960), torch.float32), ((960, 1280), torch.float32)]),
    (Matmul0, [((1, 576), torch.float32), ((576, 1024), torch.float32)]),
    (Matmul0, [((1, 1, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((1, 512, 1024), torch.float32), ((1024, 322), torch.float32)]),
    (Matmul0, [((3025, 322), torch.float32), ((322, 322), torch.float32)]),
    (Matmul0, [((1, 512, 322), torch.float32), ((1, 322, 3025), torch.float32)]),
    (Matmul0, [((1, 512, 3025), torch.float32), ((1, 3025, 322), torch.float32)]),
    (Matmul0, [((1, 512, 322), torch.float32), ((322, 1024), torch.float32)]),
    (Matmul0, [((1, 512, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((512, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((8, 512, 128), torch.float32), ((8, 128, 512), torch.float32)]),
    (Matmul0, [((8, 512, 512), torch.float32), ((8, 512, 128), torch.float32)]),
    (Matmul0, [((1, 1, 1024), torch.float32), ((1, 1024, 512), torch.float32)]),
    (Matmul0, [((1, 1, 512), torch.float32), ((1, 512, 1024), torch.float32)]),
    (Matmul0, [((1, 1, 1024), torch.float32), ((1024, 1000), torch.float32)]),
    (Matmul0, [((1, 512, 1024), torch.float32), ((1024, 512), torch.float32)]),
    (Matmul0, [((1, 50176, 256), torch.float32), ((256, 256), torch.float32)]),
    (Matmul0, [((50176, 512), torch.float32), ((512, 512), torch.float32)]),
    (Matmul0, [((1, 512, 512), torch.float32), ((1, 512, 50176), torch.float32)]),
    (Matmul0, [((1, 512, 50176), torch.float32), ((1, 50176, 512), torch.float32)]),
    (Matmul0, [((1, 512, 512), torch.float32), ((512, 1024), torch.float32)]),
    (Matmul0, [((1, 512, 1024), torch.float32), ((1024, 261), torch.float32)]),
    (Matmul0, [((50176, 261), torch.float32), ((261, 261), torch.float32)]),
    (Matmul0, [((1, 512, 261), torch.float32), ((1, 261, 50176), torch.float32)]),
    (Matmul0, [((1, 512, 50176), torch.float32), ((1, 50176, 261), torch.float32)]),
    (Matmul0, [((1, 512, 261), torch.float32), ((261, 1024), torch.float32)]),
    (Matmul0, [((1, 4096), torch.float32), ((4096, 2), torch.float32)]),
    (Matmul0, [((1, 1088), torch.float32), ((1088, 1000), torch.float32)]),
    (Matmul0, [((1, 16384, 32), torch.float32), ((32, 32), torch.float32)]),
    (Matmul0, [((256, 32), torch.float32), ((32, 32), torch.float32)]),
    (Matmul0, [((1, 16384, 32), torch.float32), ((1, 32, 256), torch.float32)]),
    (Matmul0, [((1, 16384, 256), torch.float32), ((1, 256, 32), torch.float32)]),
    (Matmul0, [((1, 16384, 32), torch.float32), ((32, 128), torch.float32)]),
    (Matmul0, [((1, 16384, 128), torch.float32), ((128, 32), torch.float32)]),
    (Matmul0, [((1, 4096, 64), torch.float32), ((64, 64), torch.float32)]),
    (Matmul0, [((256, 64), torch.float32), ((64, 64), torch.float32)]),
    (Matmul0, [((2, 4096, 32), torch.float32), ((2, 32, 256), torch.float32)]),
    (Matmul0, [((2, 4096, 256), torch.float32), ((2, 256, 32), torch.float32)]),
    (Matmul0, [((4096, 64), torch.float32), ((64, 64), torch.float32)]),
    (Matmul0, [((1, 4096, 64), torch.float32), ((64, 256), torch.float32)]),
    (Matmul0, [((1, 4096, 256), torch.float32), ((256, 64), torch.float32)]),
    (Matmul0, [((1, 1024, 160), torch.float32), ((160, 160), torch.float32)]),
    (Matmul0, [((256, 160), torch.float32), ((160, 160), torch.float32)]),
    (Matmul0, [((5, 1024, 32), torch.float32), ((5, 32, 256), torch.float32)]),
    (Matmul0, [((5, 1024, 256), torch.float32), ((5, 256, 32), torch.float32)]),
    (Matmul0, [((1024, 160), torch.float32), ((160, 160), torch.float32)]),
    (Matmul0, [((1, 1024, 160), torch.float32), ((160, 640), torch.float32)]),
    (Matmul0, [((1, 1024, 640), torch.float32), ((640, 160), torch.float32)]),
    (Matmul0, [((256, 256), torch.float32), ((256, 256), torch.float32)]),
    (Matmul0, [((8, 256, 32), torch.float32), ((8, 32, 256), torch.float32)]),
    (Matmul0, [((8, 256, 256), torch.float32), ((8, 256, 32), torch.float32)]),
    (Matmul0, [((1, 256, 256), torch.float32), ((256, 1024), torch.float32)]),
    (Matmul0, [((1, 256, 1024), torch.float32), ((1024, 256), torch.float32)]),
    (Matmul0, [((1, 256, 256), torch.float32), ((256, 256), torch.float32)]),
    (Matmul0, [((1, 1024, 160), torch.float32), ((160, 256), torch.float32)]),
    (Matmul0, [((1, 16384, 32), torch.float32), ((32, 256), torch.float32)]),
    (Matmul0, [((1, 16384, 64), torch.float32), ((64, 64), torch.float32)]),
    (Matmul0, [((1, 16384, 64), torch.float32), ((1, 64, 256), torch.float32)]),
    (Matmul0, [((1, 16384, 256), torch.float32), ((1, 256, 64), torch.float32)]),
    (Matmul0, [((1, 16384, 64), torch.float32), ((64, 256), torch.float32)]),
    (Matmul0, [((1, 16384, 256), torch.float32), ((256, 64), torch.float32)]),
    (Matmul0, [((1, 4096, 128), torch.float32), ((128, 128), torch.float32)]),
    (Matmul0, [((256, 128), torch.float32), ((128, 128), torch.float32)]),
    (Matmul0, [((2, 4096, 64), torch.float32), ((2, 64, 256), torch.float32)]),
    (Matmul0, [((2, 4096, 256), torch.float32), ((2, 256, 64), torch.float32)]),
    (Matmul0, [((4096, 128), torch.float32), ((128, 128), torch.float32)]),
    (Matmul0, [((1, 4096, 128), torch.float32), ((128, 512), torch.float32)]),
    (Matmul0, [((1, 4096, 512), torch.float32), ((512, 128), torch.float32)]),
    (Matmul0, [((1, 1024, 320), torch.float32), ((320, 320), torch.float32)]),
    (Matmul0, [((256, 320), torch.float32), ((320, 320), torch.float32)]),
    (Matmul0, [((5, 1024, 64), torch.float32), ((5, 64, 256), torch.float32)]),
    (Matmul0, [((5, 1024, 256), torch.float32), ((5, 256, 64), torch.float32)]),
    (Matmul0, [((1024, 320), torch.float32), ((320, 320), torch.float32)]),
    (Matmul0, [((1, 1024, 320), torch.float32), ((320, 1280), torch.float32)]),
    (Matmul0, [((1, 1024, 1280), torch.float32), ((1280, 320), torch.float32)]),
    (Matmul0, [((256, 512), torch.float32), ((512, 512), torch.float32)]),
    (Matmul0, [((8, 256, 64), torch.float32), ((8, 64, 256), torch.float32)]),
    (Matmul0, [((8, 256, 256), torch.float32), ((8, 256, 64), torch.float32)]),
    (Matmul0, [((1, 256, 512), torch.float32), ((512, 2048), torch.float32)]),
    (Matmul0, [((1, 256, 2048), torch.float32), ((2048, 512), torch.float32)]),
    (Matmul0, [((1, 256, 512), torch.float32), ((512, 768), torch.float32)]),
    (Matmul0, [((1, 1024, 320), torch.float32), ((320, 768), torch.float32)]),
    (Matmul0, [((1, 4096, 128), torch.float32), ((128, 768), torch.float32)]),
    (Matmul0, [((1, 16384, 64), torch.float32), ((64, 768), torch.float32)]),
    (Matmul0, [((1, 256, 512), torch.float32), ((512, 256), torch.float32)]),
    (Matmul0, [((1, 1024, 320), torch.float32), ((320, 256), torch.float32)]),
    (Matmul0, [((1, 4096, 128), torch.float32), ((128, 256), torch.float32)]),
    (Matmul0, [((1, 256), torch.float32), ((256, 1000), torch.float32)]),
    (Matmul0, [((1, 25088), torch.float32), ((25088, 4096), torch.float32)]),
    (Matmul0, [((197, 1024), torch.float32), ((1024, 1024), torch.float32)]),
    (Matmul0, [((16, 197, 64), torch.float32), ((16, 64, 197), torch.float32)]),
    (Matmul0, [((16, 197, 197), torch.float32), ((16, 197, 64), torch.float32)]),
    (Matmul0, [((1, 197, 1024), torch.float32), ((1024, 4096), torch.float32)]),
    (Matmul0, [((1, 197, 4096), torch.float32), ((4096, 1024), torch.float32)]),
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
