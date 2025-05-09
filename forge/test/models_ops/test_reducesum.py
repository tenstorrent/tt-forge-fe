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


class Reducesum0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reducesum_input_0):
        reducesum_output_1 = forge.op.ReduceSum("", reducesum_input_0, dim=-1, keep_dim=True)
        return reducesum_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reducesum0,
        [((1, 12, 8, 8), torch.float32)],
        {
            "model_names": [
                "pd_blip_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducesum0,
        [((1, 12, 11, 11), torch.float32)],
        {
            "model_names": [
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducesum0,
        [((1, 12, 9, 9), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducesum0,
        [((1, 12, 15, 15), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducesum0,
        [((1, 12, 14, 14), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
    (
        Reducesum0,
        [((1, 12, 10, 10), torch.float32)],
        {
            "model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-1", "keep_dim": "True"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("ReduceSum")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "args":
            forge_property_recorder.record_forge_op_args(metadata_value)
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

    forge_property_recorder.record_single_op_operands_info(framework_model, inputs)

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
