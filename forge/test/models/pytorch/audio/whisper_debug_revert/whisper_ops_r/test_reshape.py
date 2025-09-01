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


class Reshape0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1))
        return reshape_output_1


class Reshape1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768))
        return reshape_output_1


class Reshape2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 12, 64))
        return reshape_output_1


class Reshape3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1, 768))
        return reshape_output_1


class Reshape4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 64))
        return reshape_output_1


class Reshape5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1))
        return reshape_output_1


class Reshape6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1))
        return reshape_output_1


class Reshape7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 64))
        return reshape_output_1


class Reshape8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 80, 3000, 1))
        return reshape_output_1


class Reshape9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 80, 3, 1))
        return reshape_output_1


class Reshape10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000))
        return reshape_output_1


class Reshape11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 3000, 1))
        return reshape_output_1


class Reshape12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(768, 768, 3, 1))
        return reshape_output_1


class Reshape13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 768, 1500))
        return reshape_output_1


class Reshape14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1500, 768))
        return reshape_output_1


class Reshape15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 12, 64))
        return reshape_output_1


class Reshape16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 1500, 768))
        return reshape_output_1


class Reshape17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 64))
        return reshape_output_1


class Reshape18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 1500))
        return reshape_output_1


class Reshape19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1500, 1500))
        return reshape_output_1


class Reshape20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1500, 64))
        return reshape_output_1


class Reshape21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(1, 12, 1, 1500))
        return reshape_output_1


class Reshape22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reshape_input_0):
        reshape_output_1 = forge.op.Reshape("", reshape_input_0, shape=(12, 1, 1500))
        return reshape_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reshape0,
        [((1, 1), torch.int64)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1)"},
        },
    ),
    (
        Reshape1,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape2,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape3,
        [((1, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 768)"},
        },
    ),
    (
        Reshape2,
        [((1, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1, 12, 64)"},
        },
    ),
    (
        Reshape4,
        [((1, 12, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 64)"},
        },
    ),
    (
        Reshape5,
        [((12, 1, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 1)"},
        },
    ),
    (
        Reshape6,
        [((1, 12, 1, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 1)"},
        },
    ),
    (
        Reshape7,
        [((12, 1, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 64)"},
        },
    ),
    (
        Reshape1,
        [((1, 1, 12, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768)"},
        },
    ),
    (
        Reshape8,
        [((1, 80, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 80, 3000, 1)"},
        },
    ),
    (
        Reshape9,
        [((768, 80, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 80, 3, 1)"},
        },
    ),
    (
        Reshape10,
        [((1, 768, 3000, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 3000)"},
        },
    ),
    (
        Reshape11,
        [((1, 768, 3000), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 3000, 1)"},
        },
    ),
    (
        Reshape12,
        [((768, 768, 3), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(768, 768, 3, 1)"},
        },
    ),
    (
        Reshape13,
        [((1, 768, 1500, 1), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 768, 1500)"},
        },
    ),
    (
        Reshape14,
        [((1, 1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape15,
        [((1, 1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape16,
        [((1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 768)"},
        },
    ),
    (
        Reshape15,
        [((1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 1500, 12, 64)"},
        },
    ),
    (
        Reshape17,
        [((1, 12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1500, 64)"},
        },
    ),
    (
        Reshape18,
        [((12, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1500, 1500)"},
        },
    ),
    (
        Reshape19,
        [((1, 12, 1500, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1500, 1500)"},
        },
    ),
    (
        Reshape20,
        [((12, 1500, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1500, 64)"},
        },
    ),
    (
        Reshape14,
        [((1, 1500, 12, 64), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1500, 768)"},
        },
    ),
    (
        Reshape21,
        [((12, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(1, 12, 1, 1500)"},
        },
    ),
    (
        Reshape22,
        [((1, 12, 1, 1500), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"shape": "(12, 1, 1500)"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Reshape")

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
