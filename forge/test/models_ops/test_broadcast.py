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


class Broadcast0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-1, shape=4096)
        return broadcast_output_1


class Broadcast1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-3, shape=12)
        return broadcast_output_1


class Broadcast2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-4, shape=1)
        return broadcast_output_1


class Broadcast3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-2, shape=384)
        return broadcast_output_1


class Broadcast4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, broadcast_input_0):
        broadcast_output_1 = forge.op.Broadcast("", broadcast_input_0, dim=-2, shape=128)
        return broadcast_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    pytest.param(
        (
            Broadcast0,
            [((1, 596, 1), torch.bool)],
            {
                "model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"],
                "pcc": 0.99,
                "op_params": {"dim": "-1", "shape": "4096"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast1,
            [((1, 1, 1, 384), torch.bool)],
            {
                "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
                "pcc": 0.99,
                "op_params": {"dim": "-3", "shape": "12"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    (
        Broadcast2,
        [((1, 1, 1, 384), torch.bool)],
        {
            "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-4", "shape": "1"},
        },
    ),
    pytest.param(
        (
            Broadcast3,
            [((1, 12, 1, 384), torch.bool)],
            {
                "model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
                "pcc": 0.99,
                "op_params": {"dim": "-2", "shape": "384"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    pytest.param(
        (
            Broadcast1,
            [((1, 1, 1, 128), torch.bool)],
            {
                "model_name": [
                    "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                    "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                    "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                    "pt_distilbert_distilbert_base_cased_mlm_hf",
                    "pt_distilbert_distilbert_base_uncased_mlm_hf",
                ],
                "pcc": 0.99,
                "op_params": {"dim": "-3", "shape": "12"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
    (
        Broadcast2,
        [((1, 1, 1, 128), torch.bool)],
        {
            "model_name": [
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-4", "shape": "1"},
        },
    ),
    pytest.param(
        (
            Broadcast4,
            [((1, 12, 1, 128), torch.bool)],
            {
                "model_name": [
                    "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                    "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                    "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                    "pt_distilbert_distilbert_base_cased_mlm_hf",
                    "pt_distilbert_distilbert_base_uncased_mlm_hf",
                ],
                "pcc": 0.99,
                "op_params": {"dim": "-2", "shape": "128"},
            },
        ),
        marks=[pytest.mark.xfail(reason="RuntimeError: Generated MLIR module failed verification.")],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Broadcast")

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
