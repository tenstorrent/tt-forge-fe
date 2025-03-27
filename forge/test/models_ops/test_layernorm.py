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


class Layernorm0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm0.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm0.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm0.weight_1"),
            self.get_parameter("layernorm0.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm1.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm1.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm1.weight_1"),
            self.get_parameter("layernorm1.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm2.weight_1",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm2.weight_2",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm2.weight_1"),
            self.get_parameter("layernorm2.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm3.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm3.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm3.weight_1"),
            self.get_parameter("layernorm3.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm4.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm4.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm4.weight_1"),
            self.get_parameter("layernorm4.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm5.weight_1",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm5.weight_2",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm5.weight_1"),
            self.get_parameter("layernorm5.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm6.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm6.weight_2",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm6.weight_1"),
            self.get_parameter("layernorm6.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm7.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm7.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm7.weight_1"),
            self.get_parameter("layernorm7.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm8.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm8.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm8.weight_1"),
            self.get_parameter("layernorm8.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm9.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm9.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm9.weight_1"),
            self.get_parameter("layernorm9.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm10.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm10.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm10.weight_1"),
            self.get_parameter("layernorm10.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm11.weight_1",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm11.weight_2",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm11.weight_1"),
            self.get_parameter("layernorm11.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm12.weight_1",
            forge.Parameter(*(4544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm12.weight_2",
            forge.Parameter(*(4544,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm12.weight_1"),
            self.get_parameter("layernorm12.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm13.weight_1",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm13.weight_2",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm13.weight_1"),
            self.get_parameter("layernorm13.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm14.weight_1",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm14.weight_2",
            forge.Parameter(*(64,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm14.weight_1"),
            self.get_parameter("layernorm14.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm15.weight_1",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm15.weight_2",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm15.weight_1"),
            self.get_parameter("layernorm15.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm16.weight_1",
            forge.Parameter(*(261,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm16.weight_2",
            forge.Parameter(*(261,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm16.weight_1"),
            self.get_parameter("layernorm16.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm17.weight_1",
            forge.Parameter(*(322,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm17.weight_2",
            forge.Parameter(*(322,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm17.weight_1"),
            self.get_parameter("layernorm17.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm18.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm18.weight_2",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm18.weight_1"),
            self.get_parameter("layernorm18.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm19.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm19.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm19.weight_1"),
            self.get_parameter("layernorm19.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm20.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm20.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm20.weight_1"),
            self.get_parameter("layernorm20.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm21.weight_1",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm21.weight_2",
            forge.Parameter(*(320,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm21.weight_1"),
            self.get_parameter("layernorm21.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm22.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm22.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm22.weight_1"),
            self.get_parameter("layernorm22.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm23.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm23.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm23.weight_1"),
            self.get_parameter("layernorm23.weight_2"),
            dim=-1,
            epsilon=1e-06,
        )
        return layernorm_output_1


class Layernorm24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm24.weight_1",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm24.weight_2",
            forge.Parameter(*(32,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm24.weight_1"),
            self.get_parameter("layernorm24.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm25.weight_1",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm25.weight_2",
            forge.Parameter(*(160,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm25.weight_1"),
            self.get_parameter("layernorm25.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm26.weight_1",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm26.weight_2",
            forge.Parameter(*(256,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm26.weight_1"),
            self.get_parameter("layernorm26.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm27.weight_1",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm27.weight_2",
            forge.Parameter(*(96,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm27.weight_1"),
            self.get_parameter("layernorm27.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm28.weight_1",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm28.weight_2",
            forge.Parameter(*(192,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm28.weight_1"),
            self.get_parameter("layernorm28.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Layernorm0,
            [((2, 1, 2048), torch.float32)],
            {
                "model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"],
                "pcc": 0.99,
                "op_params": {"dim": "-1", "epsilon": "1e-05"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Layernorm1,
            [((2, 1, 1024), torch.float32)],
            {
                "model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"],
                "pcc": 0.99,
                "op_params": {"dim": "-1", "epsilon": "1e-05"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Layernorm2,
            [((2, 1, 1536), torch.float32)],
            {
                "model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
                "pcc": 0.99,
                "op_params": {"dim": "-1", "epsilon": "1e-05"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Layernorm3,
        [((1, 1, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 1500, 384), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 1500, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 1500, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm5,
        [((1, 1, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm5,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 1500, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm5,
        [((1, 2, 1280), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((2, 7, 512), torch.float32)],
        {
            "model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 577, 1024), torch.float32)],
        {
            "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm7,
        [((1, 204, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 201, 768), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm2,
        [((1, 1536), torch.float32)],
        {
            "model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm8,
        [((1, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm9,
        [((1, 128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm10,
        [((1, 128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm11,
        [((1, 128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm4,
        [((1, 128, 768), torch.float32)],
        {
            "model_name": [
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm8,
        [((1, 9, 128), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 9, 768), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm8,
        [((1, 14, 128), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 14, 768), torch.float32)],
        {
            "model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm1,
        [((1, 256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm10,
        [((1, 384, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 6, 768), torch.float32)],
        {
            "model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm2,
        [((1, 32, 1536), torch.float32)],
        {
            "model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm7,
        [((1, 384, 768), torch.float32)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm12,
        [((1, 6, 4544), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm13,
        [((1, 334, 4096), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm14,
        [((1, 334, 64, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99, "op_params": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm4,
        [((1, 256, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 7, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 32, 2560), torch.float32)],
        {
            "model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 32, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 256, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm0,
        [((1, 32, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((32, 768), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_opt_facebook_opt_125m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm0,
        [((32, 2048), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf", "pt_opt_facebook_opt_1_3b_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm0,
        [((256, 2048), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 32, 1024), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((256, 768), torch.float32)],
        {
            "model_name": ["pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm16,
        [((1, 50176, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm17,
        [((1, 3025, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 50176, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 2048, 768), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm5,
        [((1, 256, 1280), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 12, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 11, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm10,
        [((1, 197, 1024), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm10,
        [((1, 1024), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 197, 768), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 768), torch.float32)],
        {
            "model_name": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm18,
        [((1, 197, 192), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm19,
        [((1, 197, 384), torch.float32)],
        {
            "model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm14,
        [((1, 19200, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 300, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 4800, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 300, 128), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm21,
        [((1, 1200, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm21,
        [((1, 300, 320), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 300, 512), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm22,
        [((1, 196, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm4,
        [((1, 196, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 256, 512), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm23,
        [((1, 196, 1024), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-06"},
        },
    ),
    (
        Layernorm24,
        [((1, 16384, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm24,
        [((1, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 4096, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm25,
        [((1, 1024, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm25,
        [((1, 256, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm26,
        [((1, 256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm14,
        [((1, 16384, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 4096, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm21,
        [((1, 1024, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm21,
        [((1, 256, 320), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((1, 56, 56, 96), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 28, 28, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm28,
        [((1, 28, 28, 192), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 14, 14, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 14, 14, 384), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm2,
        [((1, 7, 7, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 7, 7, 768), torch.float32)],
        {
            "model_name": ["pt_swin_swin_t_img_cls_torchvision", "pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((1, 3136, 96), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 784, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm28,
        [((1, 784, 192), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 196, 384), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm2,
        [((1, 49, 1536), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 49, 768), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 56, 56, 128), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 28, 28, 512), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm26,
        [((1, 28, 28, 256), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 14, 14, 1024), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 14, 14, 512), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm0,
        [((1, 7, 7, 2048), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 7, 7, 1024), torch.float32)],
        {
            "model_name": ["pt_swin_swin_b_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Layernorm")

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
