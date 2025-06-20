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


class Reducemax0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reducemax_input_0):
        reducemax_output_1 = forge.op.ReduceMax("", reducemax_input_0, dim=-1, keep_dim=True)
        return reducemax_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reducemax0,
        [((1, 12, 11, 11), torch.float32)],
        {
            "model_names": [
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducemax0,
        [((1, 12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducemax0,
        [((1, 12, 8, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    pytest.param(
        (
            Reducemax0,
            [((1, 25, 97), torch.float32)],
            {
                "model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"],
                "pcc": 0.99,
                "args": {"dim": "-1", "keep_dim": "True"},
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Reducemax0,
        [((1, 12, 10, 10), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducemax0,
        [((1, 12, 14, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducemax0,
        [((1, 8, 12, 12), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    pytest.param(
        (
            Reducemax0,
            [((1, 12, 6625), torch.float32)],
            {
                "model_names": ["pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels"],
                "pcc": 0.99,
                "args": {"dim": "-1", "keep_dim": "True"},
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    (
        Reducemax0,
        [((1, 12, 15, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    pytest.param(
        (
            Reducemax0,
            [((1, 25, 6625), torch.float32)],
            {
                "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
                "pcc": 0.99,
                "args": {"dim": "-1", "keep_dim": "True"},
            },
        ),
        marks=[pytest.mark.xfail(reason="AssertionError: PCC is nan, but tensors are not equal")],
    ),
    (
        Reducemax0,
        [((1, 12, 97), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("ReduceMax")

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

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
