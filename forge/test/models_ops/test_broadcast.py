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


class Broadcast0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-1, shape=32)
        return broadcast_output_1


class Broadcast1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-3, shape=512)
        return broadcast_output_1


class Broadcast2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-3, shape=4)
        return broadcast_output_1


class Broadcast3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-2, shape=128)
        return broadcast_output_1


class Broadcast4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-2, shape=6)
        return broadcast_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Broadcast0,
            [((64, 4, 64, 1), torch.float32)],
            {
                "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {"dim": "-1", "shape": "32"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast0,
            [((16, 8, 64, 1), torch.float32)],
            {
                "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {"dim": "-1", "shape": "32"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast0,
            [((4, 16, 64, 1), torch.float32)],
            {
                "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {"dim": "-1", "shape": "32"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast0,
            [((1, 32, 64, 1), torch.float32)],
            {
                "model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"],
                "pcc": 0.99,
                "args": {"dim": "-1", "shape": "32"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast0,
            [((64, 3, 64, 1), torch.float32)],
            {
                "model_names": [
                    "pt_swin_swin_v2_s_img_cls_torchvision",
                    "pt_swin_swin_v2_t_img_cls_torchvision",
                    "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                ],
                "pcc": 0.99,
                "args": {"dim": "-1", "shape": "32"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast0,
            [((16, 6, 64, 1), torch.float32)],
            {
                "model_names": [
                    "pt_swin_swin_v2_s_img_cls_torchvision",
                    "pt_swin_swin_v2_t_img_cls_torchvision",
                    "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                ],
                "pcc": 0.99,
                "args": {"dim": "-1", "shape": "32"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast0,
            [((4, 12, 64, 1), torch.float32)],
            {
                "model_names": [
                    "pt_swin_swin_v2_s_img_cls_torchvision",
                    "pt_swin_swin_v2_t_img_cls_torchvision",
                    "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                ],
                "pcc": 0.99,
                "args": {"dim": "-1", "shape": "32"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast0,
            [((1, 24, 64, 1), torch.float32)],
            {
                "model_names": [
                    "pt_swin_swin_v2_s_img_cls_torchvision",
                    "pt_swin_swin_v2_t_img_cls_torchvision",
                    "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                ],
                "pcc": 0.99,
                "args": {"dim": "-1", "shape": "32"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast1,
            [((1, 1, 80, 80), torch.bfloat16)],
            {
                "model_names": ["pt_yolo_world_default_obj_det_github"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {"dim": "-3", "shape": "512"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast1,
            [((1, 1, 40, 40), torch.bfloat16)],
            {
                "model_names": ["pt_yolo_world_default_obj_det_github"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {"dim": "-3", "shape": "512"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast1,
            [((1, 1, 20, 20), torch.bfloat16)],
            {
                "model_names": ["pt_yolo_world_default_obj_det_github"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {"dim": "-3", "shape": "512"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast2,
            [((1, 8, 1, 128, 64), torch.float32)],
            {
                "model_names": [
                    "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                    "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                ],
                "pcc": 0.99,
                "args": {"dim": "-3", "shape": "4"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast3,
            [((1, 1, 1, 128), torch.int64)],
            {
                "model_names": [
                    "onnx_llama3_meta_llama_llama_3_2_1b_clm_hf",
                    "onnx_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                ],
                "pcc": 0.99,
                "args": {"dim": "-2", "shape": "128"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast4,
            [((1, 1, 1, 6), torch.int64)],
            {
                "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
                "pcc": 0.99,
                "args": {"dim": "-2", "shape": "6"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Broadcast")

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
