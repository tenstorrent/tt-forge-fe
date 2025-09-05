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


class Stack0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, stack_input_0, stack_input_1):
        stack_output_1 = forge.op.Stack("", stack_input_0, stack_input_1, axis=-1)
        return stack_output_1


class Stack1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, stack_input_0, stack_input_1, stack_input_2, stack_input_3):
        stack_output_1 = forge.op.Stack("", stack_input_0, stack_input_1, stack_input_2, stack_input_3, axis=-3)
        return stack_output_1


class Stack2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, stack_input_0):
        stack_output_1 = forge.op.Stack("", stack_input_0, axis=-4)
        return stack_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Stack0,
        [((1, 256, 16, 16), torch.float32), ((1, 256, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"axis": "-1"},
        },
    ),
    (
        Stack1,
        [
            ((1, 1, 2048), torch.float32),
            ((1, 1, 2048), torch.float32),
            ((1, 1, 2048), torch.float32),
            ((1, 1, 2048), torch.float32),
        ],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Stack0,
        [((1, 5, 16, 16), torch.float32), ((1, 5, 16, 16), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"axis": "-1"},
        },
    ),
    (
        Stack0,
        [((1, 25, 34, 64), torch.bfloat16), ((1, 25, 34, 64), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"axis": "-1"},
        },
    ),
    (
        Stack2,
        [((3, 300, 300), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"axis": "-4"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Stack")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc"]:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad=training_test)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)

    verify(
        inputs,
        framework_model,
        compiled_model,
        with_backward=training_test,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
