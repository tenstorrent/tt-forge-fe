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


class Relu0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, relu_input_0):
        relu_output_1 = forge.op.Relu("", relu_input_0)
        return relu_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Relu0, [((2, 13, 3072), torch.float32)]),
    (Relu0, [((1, 768), torch.float32)]),
    (Relu0, [((1, 334, 16384), torch.float32)]),
    (Relu0, [((32, 8192), torch.float32)]),
    (Relu0, [((256, 8192), torch.float32)]),
    (Relu0, [((32, 4096), torch.float32)]),
    (Relu0, [((32, 3072), torch.float32)]),
    (Relu0, [((256, 3072), torch.float32)]),
    (Relu0, [((256, 4096), torch.float32)]),
    (Relu0, [((1, 1, 3072), torch.float32)]),
    (Relu0, [((1, 1, 4096), torch.float32)]),
    (Relu0, [((1, 1, 2048), torch.float32)]),
    (Relu0, [((1024, 512), torch.float32)]),
    (Relu0, [((1024, 256), torch.float32)]),
    (Relu0, [((1024, 2048), torch.float32)]),
    (Relu0, [((1, 64, 55, 55), torch.float32)]),
    (Relu0, [((1, 192, 27, 27), torch.float32)]),
    (Relu0, [((1, 384, 13, 13), torch.float32)]),
    (Relu0, [((1, 256, 13, 13), torch.float32)]),
    (Relu0, [((1, 4096), torch.float32)]),
    (Relu0, [((1, 16, 28, 28), torch.float32)]),
    (Relu0, [((1, 4, 14, 14), torch.float32)]),
    (Relu0, [((1, 16, 14, 14), torch.float32)]),
    (Relu0, [((1, 128), torch.float32)]),
    (Relu0, [((1, 64), torch.float32)]),
    (Relu0, [((1, 12), torch.float32)]),
    (Relu0, [((1, 64, 112, 112), torch.float32)]),
    (Relu0, [((1, 64, 56, 56), torch.float32)]),
    (Relu0, [((1, 128, 56, 56), torch.float32)]),
    (Relu0, [((1, 96, 56, 56), torch.float32)]),
    (Relu0, [((1, 160, 56, 56), torch.float32)]),
    (Relu0, [((1, 192, 56, 56), torch.float32)]),
    (Relu0, [((1, 224, 56, 56), torch.float32)]),
    (Relu0, [((1, 256, 56, 56), torch.float32)]),
    (Relu0, [((1, 128, 28, 28), torch.float32)]),
    (Relu0, [((1, 160, 28, 28), torch.float32)]),
    (Relu0, [((1, 192, 28, 28), torch.float32)]),
    (Relu0, [((1, 224, 28, 28), torch.float32)]),
    (Relu0, [((1, 256, 28, 28), torch.float32)]),
    (Relu0, [((1, 288, 28, 28), torch.float32)]),
    (Relu0, [((1, 320, 28, 28), torch.float32)]),
    (Relu0, [((1, 352, 28, 28), torch.float32)]),
    (Relu0, [((1, 384, 28, 28), torch.float32)]),
    (Relu0, [((1, 416, 28, 28), torch.float32)]),
    (Relu0, [((1, 448, 28, 28), torch.float32)]),
    (Relu0, [((1, 480, 28, 28), torch.float32)]),
    (Relu0, [((1, 512, 28, 28), torch.float32)]),
    (Relu0, [((1, 256, 14, 14), torch.float32)]),
    (Relu0, [((1, 128, 14, 14), torch.float32)]),
    (Relu0, [((1, 288, 14, 14), torch.float32)]),
    (Relu0, [((1, 320, 14, 14), torch.float32)]),
    (Relu0, [((1, 352, 14, 14), torch.float32)]),
    (Relu0, [((1, 384, 14, 14), torch.float32)]),
    (Relu0, [((1, 416, 14, 14), torch.float32)]),
    (Relu0, [((1, 448, 14, 14), torch.float32)]),
    (Relu0, [((1, 480, 14, 14), torch.float32)]),
    (Relu0, [((1, 512, 14, 14), torch.float32)]),
    (Relu0, [((1, 544, 14, 14), torch.float32)]),
    (Relu0, [((1, 576, 14, 14), torch.float32)]),
    (Relu0, [((1, 608, 14, 14), torch.float32)]),
    (Relu0, [((1, 640, 14, 14), torch.float32)]),
    (Relu0, [((1, 672, 14, 14), torch.float32)]),
    (Relu0, [((1, 704, 14, 14), torch.float32)]),
    (Relu0, [((1, 736, 14, 14), torch.float32)]),
    (Relu0, [((1, 768, 14, 14), torch.float32)]),
    (Relu0, [((1, 800, 14, 14), torch.float32)]),
    (Relu0, [((1, 832, 14, 14), torch.float32)]),
    (Relu0, [((1, 864, 14, 14), torch.float32)]),
    (Relu0, [((1, 896, 14, 14), torch.float32)]),
    (Relu0, [((1, 928, 14, 14), torch.float32)]),
    (Relu0, [((1, 960, 14, 14), torch.float32)]),
    (Relu0, [((1, 992, 14, 14), torch.float32)]),
    (Relu0, [((1, 1024, 14, 14), torch.float32)]),
    (Relu0, [((1, 1056, 14, 14), torch.float32)]),
    (Relu0, [((1, 1088, 14, 14), torch.float32)]),
    (Relu0, [((1, 1120, 14, 14), torch.float32)]),
    (Relu0, [((1, 1152, 14, 14), torch.float32)]),
    (Relu0, [((1, 1184, 14, 14), torch.float32)]),
    (Relu0, [((1, 1216, 14, 14), torch.float32)]),
    (Relu0, [((1, 1248, 14, 14), torch.float32)]),
    (Relu0, [((1, 1280, 14, 14), torch.float32)]),
    (Relu0, [((1, 1312, 14, 14), torch.float32)]),
    (Relu0, [((1, 1344, 14, 14), torch.float32)]),
    (Relu0, [((1, 1376, 14, 14), torch.float32)]),
    (Relu0, [((1, 1408, 14, 14), torch.float32)]),
    (Relu0, [((1, 1440, 14, 14), torch.float32)]),
    (Relu0, [((1, 1472, 14, 14), torch.float32)]),
    (Relu0, [((1, 1504, 14, 14), torch.float32)]),
    (Relu0, [((1, 1536, 14, 14), torch.float32)]),
    (Relu0, [((1, 1568, 14, 14), torch.float32)]),
    (Relu0, [((1, 1600, 14, 14), torch.float32)]),
    (Relu0, [((1, 1632, 14, 14), torch.float32)]),
    (Relu0, [((1, 1664, 14, 14), torch.float32)]),
    (Relu0, [((1, 1696, 14, 14), torch.float32)]),
    (Relu0, [((1, 1728, 14, 14), torch.float32)]),
    (Relu0, [((1, 1760, 14, 14), torch.float32)]),
    (Relu0, [((1, 1792, 14, 14), torch.float32)]),
    (Relu0, [((1, 896, 7, 7), torch.float32)]),
    (Relu0, [((1, 128, 7, 7), torch.float32)]),
    (Relu0, [((1, 928, 7, 7), torch.float32)]),
    (Relu0, [((1, 960, 7, 7), torch.float32)]),
    (Relu0, [((1, 992, 7, 7), torch.float32)]),
    (Relu0, [((1, 1024, 7, 7), torch.float32)]),
    (Relu0, [((1, 1056, 7, 7), torch.float32)]),
    (Relu0, [((1, 1088, 7, 7), torch.float32)]),
    (Relu0, [((1, 1120, 7, 7), torch.float32)]),
    (Relu0, [((1, 1152, 7, 7), torch.float32)]),
    (Relu0, [((1, 1184, 7, 7), torch.float32)]),
    (Relu0, [((1, 1216, 7, 7), torch.float32)]),
    (Relu0, [((1, 1248, 7, 7), torch.float32)]),
    (Relu0, [((1, 1280, 7, 7), torch.float32)]),
    (Relu0, [((1, 1312, 7, 7), torch.float32)]),
    (Relu0, [((1, 1344, 7, 7), torch.float32)]),
    (Relu0, [((1, 1376, 7, 7), torch.float32)]),
    (Relu0, [((1, 1408, 7, 7), torch.float32)]),
    (Relu0, [((1, 1440, 7, 7), torch.float32)]),
    (Relu0, [((1, 1472, 7, 7), torch.float32)]),
    (Relu0, [((1, 1504, 7, 7), torch.float32)]),
    (Relu0, [((1, 1536, 7, 7), torch.float32)]),
    (Relu0, [((1, 1568, 7, 7), torch.float32)]),
    (Relu0, [((1, 1600, 7, 7), torch.float32)]),
    (Relu0, [((1, 1632, 7, 7), torch.float32)]),
    (Relu0, [((1, 1664, 7, 7), torch.float32)]),
    (Relu0, [((1, 1696, 7, 7), torch.float32)]),
    (Relu0, [((1, 1728, 7, 7), torch.float32)]),
    (Relu0, [((1, 1760, 7, 7), torch.float32)]),
    (Relu0, [((1, 1792, 7, 7), torch.float32)]),
    (Relu0, [((1, 1824, 7, 7), torch.float32)]),
    (Relu0, [((1, 1856, 7, 7), torch.float32)]),
    (Relu0, [((1, 1888, 7, 7), torch.float32)]),
    (Relu0, [((1, 1920, 7, 7), torch.float32)]),
    (Relu0, [((1, 512, 7, 7), torch.float32)]),
    (Relu0, [((1, 544, 7, 7), torch.float32)]),
    (Relu0, [((1, 576, 7, 7), torch.float32)]),
    (Relu0, [((1, 608, 7, 7), torch.float32)]),
    (Relu0, [((1, 640, 7, 7), torch.float32)]),
    (Relu0, [((1, 672, 7, 7), torch.float32)]),
    (Relu0, [((1, 704, 7, 7), torch.float32)]),
    (Relu0, [((1, 736, 7, 7), torch.float32)]),
    (Relu0, [((1, 768, 7, 7), torch.float32)]),
    (Relu0, [((1, 800, 7, 7), torch.float32)]),
    (Relu0, [((1, 832, 7, 7), torch.float32)]),
    (Relu0, [((1, 864, 7, 7), torch.float32)]),
    (Relu0, [((1, 96, 112, 112), torch.float32)]),
    (Relu0, [((1, 144, 56, 56), torch.float32)]),
    (Relu0, [((1, 240, 56, 56), torch.float32)]),
    (Relu0, [((1, 288, 56, 56), torch.float32)]),
    (Relu0, [((1, 336, 56, 56), torch.float32)]),
    (Relu0, [((1, 384, 56, 56), torch.float32)]),
    (Relu0, [((1, 240, 28, 28), torch.float32)]),
    (Relu0, [((1, 336, 28, 28), torch.float32)]),
    (Relu0, [((1, 432, 28, 28), torch.float32)]),
    (Relu0, [((1, 528, 28, 28), torch.float32)]),
    (Relu0, [((1, 576, 28, 28), torch.float32)]),
    (Relu0, [((1, 624, 28, 28), torch.float32)]),
    (Relu0, [((1, 672, 28, 28), torch.float32)]),
    (Relu0, [((1, 720, 28, 28), torch.float32)]),
    (Relu0, [((1, 768, 28, 28), torch.float32)]),
    (Relu0, [((1, 192, 14, 14), torch.float32)]),
    (Relu0, [((1, 432, 14, 14), torch.float32)]),
    (Relu0, [((1, 528, 14, 14), torch.float32)]),
    (Relu0, [((1, 624, 14, 14), torch.float32)]),
    (Relu0, [((1, 720, 14, 14), torch.float32)]),
    (Relu0, [((1, 816, 14, 14), torch.float32)]),
    (Relu0, [((1, 912, 14, 14), torch.float32)]),
    (Relu0, [((1, 1008, 14, 14), torch.float32)]),
    (Relu0, [((1, 1104, 14, 14), torch.float32)]),
    (Relu0, [((1, 1200, 14, 14), torch.float32)]),
    (Relu0, [((1, 1296, 14, 14), torch.float32)]),
    (Relu0, [((1, 1392, 14, 14), torch.float32)]),
    (Relu0, [((1, 1488, 14, 14), torch.float32)]),
    (Relu0, [((1, 1584, 14, 14), torch.float32)]),
    (Relu0, [((1, 1680, 14, 14), torch.float32)]),
    (Relu0, [((1, 1776, 14, 14), torch.float32)]),
    (Relu0, [((1, 1824, 14, 14), torch.float32)]),
    (Relu0, [((1, 1872, 14, 14), torch.float32)]),
    (Relu0, [((1, 1920, 14, 14), torch.float32)]),
    (Relu0, [((1, 1968, 14, 14), torch.float32)]),
    (Relu0, [((1, 2016, 14, 14), torch.float32)]),
    (Relu0, [((1, 2064, 14, 14), torch.float32)]),
    (Relu0, [((1, 2112, 14, 14), torch.float32)]),
    (Relu0, [((1, 192, 7, 7), torch.float32)]),
    (Relu0, [((1, 1104, 7, 7), torch.float32)]),
    (Relu0, [((1, 1200, 7, 7), torch.float32)]),
    (Relu0, [((1, 1296, 7, 7), torch.float32)]),
    (Relu0, [((1, 1392, 7, 7), torch.float32)]),
    (Relu0, [((1, 1488, 7, 7), torch.float32)]),
    (Relu0, [((1, 1584, 7, 7), torch.float32)]),
    (Relu0, [((1, 1680, 7, 7), torch.float32)]),
    (Relu0, [((1, 1776, 7, 7), torch.float32)]),
    (Relu0, [((1, 1872, 7, 7), torch.float32)]),
    (Relu0, [((1, 1968, 7, 7), torch.float32)]),
    (Relu0, [((1, 2016, 7, 7), torch.float32)]),
    (Relu0, [((1, 2064, 7, 7), torch.float32)]),
    (Relu0, [((1, 2112, 7, 7), torch.float32)]),
    (Relu0, [((1, 2160, 7, 7), torch.float32)]),
    (Relu0, [((1, 2208, 7, 7), torch.float32)]),
    (Relu0, [((1, 16, 224, 224), torch.float32)]),
    (Relu0, [((1, 32, 112, 112), torch.float32)]),
    (Relu0, [((1, 64, 28, 28), torch.float32)]),
    (Relu0, [((1, 256, 7, 7), torch.float32)]),
    (Relu0, [((1, 32, 56, 56), torch.float32)]),
    (Relu0, [((1, 32, 28, 28), torch.float32)]),
    (Relu0, [((1, 64, 14, 14), torch.float32)]),
    (Relu0, [((1, 256, 112, 112), torch.float32)]),
    (Relu0, [((1, 512, 56, 56), torch.float32)]),
    (Relu0, [((1, 1024, 28, 28), torch.float32)]),
    (Relu0, [((1, 2048, 14, 14), torch.float32)]),
    (Relu0, [((1, 2048, 7, 7), torch.float32)]),
    (Relu0, [((1, 128, 112, 112), torch.float32)]),
    (Relu0, [((1, 16, 112, 112), torch.float32)]),
    (Relu0, [((1, 8, 112, 112), torch.float32)]),
    (Relu0, [((1, 24, 112, 112), torch.float32)]),
    (Relu0, [((1, 36, 56, 56), torch.float32)]),
    (Relu0, [((1, 20, 1, 1), torch.float32)]),
    (Relu0, [((1, 60, 28, 28), torch.float32)]),
    (Relu0, [((1, 32, 1, 1), torch.float32)]),
    (Relu0, [((1, 120, 28, 28), torch.float32)]),
    (Relu0, [((1, 100, 14, 14), torch.float32)]),
    (Relu0, [((1, 92, 14, 14), torch.float32)]),
    (Relu0, [((1, 240, 14, 14), torch.float32)]),
    (Relu0, [((1, 120, 1, 1), torch.float32)]),
    (Relu0, [((1, 336, 14, 14), torch.float32)]),
    (Relu0, [((1, 168, 1, 1), torch.float32)]),
    (Relu0, [((1, 480, 7, 7), torch.float32)]),
    (Relu0, [((1, 240, 1, 1), torch.float32)]),
    (Relu0, [((1, 1280, 1, 1), torch.float32)]),
    (Relu0, [((1, 176, 28, 28), torch.float32)]),
    (Relu0, [((1, 96, 28, 28), torch.float32)]),
    (Relu0, [((1, 304, 14, 14), torch.float32)]),
    (Relu0, [((1, 208, 14, 14), torch.float32)]),
    (Relu0, [((1, 48, 14, 14), torch.float32)]),
    (Relu0, [((1, 296, 14, 14), torch.float32)]),
    (Relu0, [((1, 224, 14, 14), torch.float32)]),
    (Relu0, [((1, 280, 14, 14), torch.float32)]),
    (Relu0, [((1, 448, 7, 7), torch.float32)]),
    (Relu0, [((1, 320, 7, 7), torch.float32)]),
    (Relu0, [((1, 624, 7, 7), torch.float32)]),
    (Relu0, [((1, 384, 7, 7), torch.float32)]),
    (Relu0, [((1, 18, 56, 56), torch.float32)]),
    (Relu0, [((1, 36, 28, 28), torch.float32)]),
    (Relu0, [((1, 72, 14, 14), torch.float32)]),
    (Relu0, [((1, 18, 28, 28), torch.float32)]),
    (Relu0, [((1, 144, 7, 7), torch.float32)]),
    (Relu0, [((1, 18, 14, 14), torch.float32)]),
    (Relu0, [((1, 36, 14, 14), torch.float32)]),
    (Relu0, [((1, 40, 56, 56), torch.float32)]),
    (Relu0, [((1, 80, 28, 28), torch.float32)]),
    (Relu0, [((1, 160, 14, 14), torch.float32)]),
    (Relu0, [((1, 40, 28, 28), torch.float32)]),
    (Relu0, [((1, 40, 14, 14), torch.float32)]),
    (Relu0, [((1, 80, 14, 14), torch.float32)]),
    (Relu0, [((1, 32, 14, 14), torch.float32)]),
    (Relu0, [((1, 16, 56, 56), torch.float32)]),
    (Relu0, [((1, 44, 56, 56), torch.float32)]),
    (Relu0, [((1, 88, 28, 28), torch.float32)]),
    (Relu0, [((1, 176, 14, 14), torch.float32)]),
    (Relu0, [((1, 44, 28, 28), torch.float32)]),
    (Relu0, [((1, 352, 7, 7), torch.float32)]),
    (Relu0, [((1, 44, 14, 14), torch.float32)]),
    (Relu0, [((1, 88, 14, 14), torch.float32)]),
    (Relu0, [((1, 48, 56, 56), torch.float32)]),
    (Relu0, [((1, 48, 28, 28), torch.float32)]),
    (Relu0, [((1, 96, 14, 14), torch.float32)]),
    (Relu0, [((1, 30, 56, 56), torch.float32)]),
    (Relu0, [((1, 120, 14, 14), torch.float32)]),
    (Relu0, [((1, 30, 28, 28), torch.float32)]),
    (Relu0, [((1, 240, 7, 7), torch.float32)]),
    (Relu0, [((1, 30, 14, 14), torch.float32)]),
    (Relu0, [((1, 60, 14, 14), torch.float32)]),
    (Relu0, [((1, 32, 149, 149), torch.float32)]),
    (Relu0, [((1, 32, 147, 147), torch.float32)]),
    (Relu0, [((1, 64, 147, 147), torch.float32)]),
    (Relu0, [((1, 96, 73, 73), torch.float32)]),
    (Relu0, [((1, 64, 73, 73), torch.float32)]),
    (Relu0, [((1, 96, 71, 71), torch.float32)]),
    (Relu0, [((1, 192, 35, 35), torch.float32)]),
    (Relu0, [((1, 224, 35, 35), torch.float32)]),
    (Relu0, [((1, 96, 35, 35), torch.float32)]),
    (Relu0, [((1, 384, 17, 17), torch.float32)]),
    (Relu0, [((1, 256, 17, 17), torch.float32)]),
    (Relu0, [((1, 768, 17, 17), torch.float32)]),
    (Relu0, [((1, 224, 17, 17), torch.float32)]),
    (Relu0, [((1, 192, 17, 17), torch.float32)]),
    (Relu0, [((1, 128, 17, 17), torch.float32)]),
    (Relu0, [((1, 192, 8, 8), torch.float32)]),
    (Relu0, [((1, 320, 17, 17), torch.float32)]),
    (Relu0, [((1, 320, 8, 8), torch.float32)]),
    (Relu0, [((1, 1024, 8, 8), torch.float32)]),
    (Relu0, [((1, 256, 8, 8), torch.float32)]),
    (Relu0, [((1, 448, 8, 8), torch.float32)]),
    (Relu0, [((1, 512, 8, 8), torch.float32)]),
    (Relu0, [((1, 32, 32, 32), torch.float32)]),
    (Relu0, [((1, 64, 32, 32), torch.float32)]),
    (Relu0, [((1, 64, 16, 16), torch.float32)]),
    (Relu0, [((1, 128, 16, 16), torch.float32)]),
    (Relu0, [((1, 128, 8, 8), torch.float32)]),
    (Relu0, [((1, 256, 4, 4), torch.float32)]),
    (Relu0, [((1, 512, 4, 4), torch.float32)]),
    (Relu0, [((1, 512, 2, 2), torch.float32)]),
    (Relu0, [((1, 1024, 2, 2), torch.float32)]),
    (Relu0, [((1, 256, 1, 1), torch.float32)]),
    (Relu0, [((1, 8, 1, 1), torch.float32)]),
    (Relu0, [((1, 72, 56, 56), torch.float32)]),
    (Relu0, [((1, 72, 28, 28), torch.float32)]),
    (Relu0, [((1, 24, 1, 1), torch.float32)]),
    (Relu0, [((1, 64, 1, 1), torch.float32)]),
    (Relu0, [((1, 40, 1, 1), torch.float32)]),
    (Relu0, [((1, 72, 1, 1), torch.float32)]),
    (Relu0, [((1, 144, 1, 1), torch.float32)]),
    (Relu0, [((1, 64, 160, 512), torch.float32)]),
    (Relu0, [((1, 64, 80, 256), torch.float32)]),
    (Relu0, [((1, 128, 40, 128), torch.float32)]),
    (Relu0, [((1, 256, 20, 64), torch.float32)]),
    (Relu0, [((1, 512, 10, 32), torch.float32)]),
    (Relu0, [((1, 64, 96, 320), torch.float32)]),
    (Relu0, [((1, 64, 48, 160), torch.float32)]),
    (Relu0, [((1, 128, 24, 80), torch.float32)]),
    (Relu0, [((1, 256, 12, 40), torch.float32)]),
    (Relu0, [((1, 512, 6, 20), torch.float32)]),
    (Relu0, [((1, 48, 1, 1), torch.float32)]),
    (Relu0, [((1, 128, 1, 1), torch.float32)]),
    (Relu0, [((1, 272, 1, 1), torch.float32)]),
    (Relu0, [((1, 64, 240, 320), torch.float32)]),
    (Relu0, [((1, 64, 120, 160), torch.float32)]),
    (Relu0, [((1, 128, 60, 80), torch.float32)]),
    (Relu0, [((1, 256, 30, 40), torch.float32)]),
    (Relu0, [((1, 512, 15, 20), torch.float32)]),
    (Relu0, [((1, 256, 60, 80), torch.float32)]),
    (Relu0, [((1, 256, 15, 20), torch.float32)]),
    (Relu0, [((1, 256, 8, 10), torch.float32)]),
    (Relu0, [((1, 256, 4, 5), torch.float32)]),
    (Relu0, [((1, 256, 120, 160), torch.float32)]),
    (Relu0, [((1, 128, 120, 160), torch.float32)]),
    (Relu0, [((1, 512, 60, 80), torch.float32)]),
    (Relu0, [((1, 1024, 30, 40), torch.float32)]),
    (Relu0, [((1, 512, 30, 40), torch.float32)]),
    (Relu0, [((1, 2048, 15, 20), torch.float32)]),
    (Relu0, [((1, 256, 128, 128), torch.float32)]),
    (Relu0, [((1, 768, 128, 128), torch.float32)]),
    (Relu0, [((1, 64, 150, 150), torch.float32)]),
    (Relu0, [((1, 64, 75, 75), torch.float32)]),
    (Relu0, [((1, 256, 75, 75), torch.float32)]),
    (Relu0, [((1, 128, 75, 75), torch.float32)]),
    (Relu0, [((1, 128, 38, 38), torch.float32)]),
    (Relu0, [((1, 512, 38, 38), torch.float32)]),
    (Relu0, [((1, 256, 38, 38), torch.float32)]),
    (Relu0, [((1, 1024, 38, 38), torch.float32)]),
    (Relu0, [((1, 512, 19, 19), torch.float32)]),
    (Relu0, [((1, 256, 19, 19), torch.float32)]),
    (Relu0, [((1, 512, 10, 10), torch.float32)]),
    (Relu0, [((1, 128, 10, 10), torch.float32)]),
    (Relu0, [((1, 256, 5, 5), torch.float32)]),
    (Relu0, [((1, 128, 5, 5), torch.float32)]),
    (Relu0, [((1, 256, 3, 3), torch.float32)]),
    (Relu0, [((1, 128, 3, 3), torch.float32)]),
    (Relu0, [((1, 32, 256, 256), torch.float32)]),
    (Relu0, [((1, 64, 128, 128), torch.float32)]),
    (Relu0, [((1, 128, 64, 64), torch.float32)]),
    (Relu0, [((1, 256, 32, 32), torch.float32)]),
    (Relu0, [((1, 512, 16, 16), torch.float32)]),
    (Relu0, [((1, 64, 224, 224), torch.float32)]),
    (Relu0, [((1, 4096, 1, 1), torch.float32)]),
    (Relu0, [((1, 224, 7, 7), torch.float32)]),
    (Relu0, [((1, 112, 7, 7), torch.float32)]),
    (Relu0, [((1, 32, 150, 150), torch.float32)]),
    (Relu0, [((1, 128, 150, 150), torch.float32)]),
    (Relu0, [((1, 728, 38, 38), torch.float32)]),
    (Relu0, [((1, 728, 19, 19), torch.float32)]),
    (Relu0, [((1, 1024, 19, 19), torch.float32)]),
    (Relu0, [((1, 1024, 10, 10), torch.float32)]),
    (Relu0, [((1, 1536, 10, 10), torch.float32)]),
    (Relu0, [((1, 2048, 10, 10), torch.float32)]),
    (Relu0, [((1, 128, 147, 147), torch.float32)]),
    (Relu0, [((1, 128, 74, 74), torch.float32)]),
    (Relu0, [((1, 256, 74, 74), torch.float32)]),
    (Relu0, [((1, 256, 37, 37), torch.float32)]),
    (Relu0, [((1, 728, 37, 37), torch.float32)]),
    (Relu0, [((1, 16, 224, 320), torch.float32)]),
    (Relu0, [((1, 32, 112, 160), torch.float32)]),
    (Relu0, [((1, 64, 56, 80), torch.float32)]),
    (Relu0, [((1, 128, 28, 40), torch.float32)]),
    (Relu0, [((1, 256, 14, 20), torch.float32)]),
    (Relu0, [((1, 128, 14, 20), torch.float32)]),
    (Relu0, [((1, 64, 14, 20), torch.float32)]),
    (Relu0, [((1, 64, 28, 40), torch.float32)]),
    (Relu0, [((1, 32, 28, 40), torch.float32)]),
    (Relu0, [((1, 32, 56, 80), torch.float32)]),
    (Relu0, [((1, 48, 224, 320), torch.float32)]),
    (Relu0, [((1, 96, 112, 160), torch.float32)]),
    (Relu0, [((1, 64, 112, 160), torch.float32)]),
    (Relu0, [((1, 192, 56, 80), torch.float32)]),
    (Relu0, [((1, 128, 56, 80), torch.float32)]),
    (Relu0, [((1, 384, 28, 40), torch.float32)]),
    (Relu0, [((1, 256, 28, 40), torch.float32)]),
    (Relu0, [((1, 768, 14, 20), torch.float32)]),
    (Relu0, [((1, 512, 14, 20), torch.float32)]),
    (Relu0, [((1, 384, 14, 20), torch.float32)]),
    (Relu0, [((1, 192, 14, 20), torch.float32)]),
    (Relu0, [((1, 192, 28, 40), torch.float32)]),
    (Relu0, [((1, 96, 28, 40), torch.float32)]),
    (Relu0, [((1, 96, 56, 80), torch.float32)]),
    (Relu0, [((1, 32, 224, 320), torch.float32)]),
    (Relu0, [((1, 256, 56, 80), torch.float32)]),
    (Relu0, [((1, 128, 112, 160), torch.float32)]),
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
