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


class Tanh0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, tanh_input_0):
        tanh_output_1 = forge.op.Tanh("", tanh_input_0)
        return tanh_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Tanh0,
        [((1, 768), torch.float32)],
        {
            "model_names": [
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Tanh0, [((1, 32, 480, 640), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Tanh0, [((1, 64, 240, 320), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Tanh0, [((1, 32, 240, 320), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (
        Tanh0,
        [((1, 128, 120, 160), torch.bfloat16)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (Tanh0, [((1, 64, 120, 160), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Tanh0, [((1, 256, 60, 80), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Tanh0, [((1, 128, 60, 80), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Tanh0, [((1, 512, 30, 40), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Tanh0, [((1, 256, 30, 40), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Tanh0, [((1, 1024, 15, 20), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (Tanh0, [((1, 512, 15, 20), torch.bfloat16)], {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99}),
    (
        Tanh0,
        [((1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (Tanh0, [((1, 32, 6144), torch.float32)], {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99}),
    (
        Tanh0,
        [((1, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Tanh0,
        [((1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf", "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"],
            "pcc": 0.99,
        },
    ),
    (Tanh0, [((1, 8, 207, 207), torch.float32)], {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99}),
    (Tanh0, [((1, 207, 256000), torch.float32)], {"model_names": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99}),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Tanh")

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
