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


class Unsqueeze0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, unsqueeze_input_0):
        unsqueeze_output_1 = forge.op.Unsqueeze("", unsqueeze_input_0, dim=1)
        return unsqueeze_output_1


class Unsqueeze1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, unsqueeze_input_0):
        unsqueeze_output_1 = forge.op.Unsqueeze("", unsqueeze_input_0, dim=2)
        return unsqueeze_output_1


class Unsqueeze2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, unsqueeze_input_0):
        unsqueeze_output_1 = forge.op.Unsqueeze("", unsqueeze_input_0, dim=0)
        return unsqueeze_output_1


class Unsqueeze3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, unsqueeze_input_0):
        unsqueeze_output_1 = forge.op.Unsqueeze("", unsqueeze_input_0, dim=-1)
        return unsqueeze_output_1


class Unsqueeze4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, unsqueeze_input_0):
        unsqueeze_output_1 = forge.op.Unsqueeze("", unsqueeze_input_0, dim=3)
        return unsqueeze_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Unsqueeze0,
        [((1, 6), torch.int64)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 1, 6), torch.int64)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze0,
        [((16,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_yolov8_default_obj_det_github",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze2,
        [((16,), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "0"},
        },
    ),
    (
        Unsqueeze0,
        [((16, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_yolov8_default_obj_det_github",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((32,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_yolov8_default_obj_det_github",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze2,
        [((32,), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "0"},
        },
    ),
    (
        Unsqueeze0,
        [((32, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_yolov8_default_obj_det_github",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((48,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((48, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((96,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((96, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((192,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((192, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((384,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((384, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((360,), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((360, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((24,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((24, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((6,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((6, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((42,), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((42, 1), torch.float32)],
        {
            "model_names": ["TranslatedLayer", "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((18,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((18, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((12,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((12, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1,), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 1), torch.float32)],
        {
            "model_names": [
                "TranslatedLayer",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((256,), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_yolov8_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((256, 1), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_yolov8_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((128,), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_yolov8_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((128, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_yolov8_default_obj_det_github",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((512,), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_whisper_openai_whisper_base_speech_recognition_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((512, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1024, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((2048,), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze2,
        [((2048,), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "0"}},
    ),
    (
        Unsqueeze0,
        [((2048, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_mamba_state_spaces_mamba_370m_hf_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1000,), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1000, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((64,), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_yolov8_default_obj_det_github",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze2,
        [((64,), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim": "0"}},
    ),
    (
        Unsqueeze0,
        [((64, 1), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pd_resnet_152_img_cls_paddlemodels",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_yolov8_default_obj_det_github",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((40,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((40, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((10,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((10, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((144,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((144, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((8,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((8, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((288,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((288, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((576,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((576, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((136,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((136, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((816,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((816, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((34,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((34, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((232,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((232, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1392,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1392, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((58,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((58, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((2304,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((2304, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1536,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1536, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b3_img_cls_timm",
                "onnx_efficientnet_efficientnet_b3a_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((320,), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((320, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1280,), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_whisper_openai_whisper_large_v3_clm_hf",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_whisper_openai_whisper_large_v3_clm_hf",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1280, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((768,), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((768, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((150,), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((150, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((80,), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((80, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_yolov8_default_obj_det_github",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((112,), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((112, 1), torch.float32)],
        {
            "model_names": [
                "onnx_vovnet_vovnet27s_obj_det_osmr",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze0,
        [((240,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((240, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((60,), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((60, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((480,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((480, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((120,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((120, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 1, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 32), torch.int64)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 1, 32), torch.int64)],
        {
            "model_names": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 32), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 8, 256, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze1,
        [((1, 16), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_clm_hf", "pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze3,
        [((1, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Unsqueeze0,
        [((1, 256, 32), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1, 204), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 1, 204), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((4,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((4, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((20,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((20, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((672,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((672, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((28,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((28, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1152,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1152, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b0_img_cls_timm",
                "onnx_efficientnet_efficientnet_b1_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((160,), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((160, 1), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv2_mobilenetv2_050_img_cls_timm",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((224,), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((224, 1), torch.float32)],
        {
            "model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((1, 11), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 1, 11), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((2, 7), torch.int64)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((2, 1, 7), torch.int64)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze1,
        [((1, 1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 8, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((1, 7, 32), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1, 201), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 1, 201), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((1, 32, 107, 160), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1, 64, 54, 80), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1, 128, 27, 40), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1, 256, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze3,
        [((1, 100, 8, 32), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Unsqueeze0,
        [((1, 8, 32, 280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((264,), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((264, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1920,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1920, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b1_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((56,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((56, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((336,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((336, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((14,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((14, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((960,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((960, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b4_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((272,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((272, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1632,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1632, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((68,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((68, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((448,), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((448, 1), torch.float32)],
        {
            "model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm", "pd_densenet_121_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((2688,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((2688, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1792,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1792, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b4_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((640,), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((640, 1), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((2,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((2, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((30,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((30, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((72,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((72, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((36,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((36, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze2,
        [((1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "0"},
        },
    ),
    (
        Unsqueeze0,
        [((25, 1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((88,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((88, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((528,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((528, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((22,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((22, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((720,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((720, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((208,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((208, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1248,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1248, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((52,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((52, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((352,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((352, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b2_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((2112,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((2112, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1408,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1408, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b2_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((176,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((176, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1056,), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((1056, 1), torch.float32)],
        {
            "model_names": [
                "onnx_efficientnet_efficientnet_b5_img_cls_timm",
                "onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Unsqueeze0,
        [((44,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((44, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((304,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((304, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1824,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1824, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((76,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((76, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((3072,), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((3072, 1), torch.float32)],
        {"model_names": ["onnx_efficientnet_efficientnet_b5_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((432,), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((432, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((104,), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((104, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((624,), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((624, 1), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze2,
        [((3, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "0"},
        },
    ),
    (
        Unsqueeze2,
        [((6, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "0"},
        },
    ),
    (
        Unsqueeze2,
        [((12, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "0"},
        },
    ),
    (
        Unsqueeze2,
        [((24, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "0"},
        },
    ),
    (
        Unsqueeze0,
        [((416,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((416, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((544,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((544, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((608,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((608, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((704,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((704, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((736,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((736, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((800,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((800, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((832,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((832, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((864,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((864, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((896,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((896, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((928,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((928, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((992,), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((992, 1), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze0,
        [((1, 14), torch.int64)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 1, 14), torch.int64)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze1,
        [((1, 64), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((1, 588, 128), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze2,
        [((2048, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "0"}},
    ),
    (
        Unsqueeze1,
        [((1, 2048, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze1,
        [((1, 2048, 6), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze4,
        [((1, 2048, 6), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Unsqueeze1,
        [((2048, 1, 4), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze0,
        [((1, 6, 16), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((1, 2048), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze2,
        [((12, 13, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim": "0"}},
    ),
    (
        Unsqueeze0,
        [((2, 13), torch.int64)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Unsqueeze1,
        [((2, 13), torch.int64)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Unsqueeze1,
        [((2, 1, 13), torch.int64)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Unsqueeze")

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
