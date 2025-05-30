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


class Reciprocal0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, reciprocal_input_0):
        reciprocal_output_1 = forge.op.Reciprocal("", reciprocal_input_0)
        return reciprocal_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Reciprocal0,
        [((16,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((24,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 12, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 11, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 12, 11, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((60,), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((120,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 8, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((480,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((128,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((2048,), torch.float32)],
        {"model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 256, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 10, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 12, 10, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 8, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 12, 8, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 4, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (Reciprocal0, [((1, 1, 1, 64), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 256), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 128), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 512), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 1024), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (Reciprocal0, [((1, 1, 1, 2048), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Reciprocal0,
        [((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 12, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((8,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((40,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((48,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((72,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((144,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((288,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1, 25, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((64, 3, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((16, 6, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((4, 12, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 24, 64, 32), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((96,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((160,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((192,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Reciprocal0, [((224,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Reciprocal0,
        [((320,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Reciprocal0, [((352,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Reciprocal0,
        [((384,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Reciprocal0, [((416,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((448,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((544,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Reciprocal0,
        [((576,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Reciprocal0, [((608,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((640,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((672,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((704,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((736,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((768,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((800,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((832,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((864,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((896,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (Reciprocal0, [((928,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Reciprocal0,
        [((960,), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (Reciprocal0, [((992,), torch.float32)], {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Reciprocal0,
        [((32,), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Reciprocal0,
        [((1280,), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 588, 1), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((1, 6, 1), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Reciprocal0,
        [((2, 13, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Reciprocal")

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

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)))
