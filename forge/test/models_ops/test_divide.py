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


class Divide0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide0_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, divide_input_0):
        divide_output_1 = forge.op.Divide("", divide_input_0, self.get_constant("divide0_const_1"))
        return divide_output_1


class Divide1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide1_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, divide_input_1):
        divide_output_1 = forge.op.Divide("", self.get_constant("divide1_const_0"), divide_input_1)
        return divide_output_1


class Divide2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, divide_input_0, divide_input_1):
        divide_output_1 = forge.op.Divide("", divide_input_0, divide_input_1)
        return divide_output_1


class Divide3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide3_const_0", shape=(1,), dtype=torch.bfloat16)

    def forward(self, divide_input_1):
        divide_output_1 = forge.op.Divide("", self.get_constant("divide3_const_0"), divide_input_1)
        return divide_output_1


class Divide4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide4_const_1", shape=(1,), dtype=torch.bfloat16)

    def forward(self, divide_input_0):
        divide_output_1 = forge.op.Divide("", divide_input_0, self.get_constant("divide4_const_1"))
        return divide_output_1


class Divide5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide5_const_1", shape=(128,), dtype=torch.bfloat16)

    def forward(self, divide_input_0):
        divide_output_1 = forge.op.Divide("", divide_input_0, self.get_constant("divide5_const_1"))
        return divide_output_1


class Divide6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("divide6_const_1", shape=(3, 1, 1), dtype=torch.bfloat16)

    def forward(self, divide_input_0):
        divide_output_1 = forge.op.Divide("", divide_input_0, self.get_constant("divide6_const_1"))
        return divide_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Divide0, [((1, 128, 16384), torch.float32)], {"model_names": ["onnx_albert_xxlarge_v1_mlm_hf"], "pcc": 0.99}),
    (
        Divide0,
        [((1, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 512, 256), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_s16_224_img_cls_timm", "onnx_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 196, 2048), torch.float32)],
        {"model_names": ["onnx_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 4096, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((16,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 16, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 32, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 64, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 64, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 128, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 128, 8, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 240, 8, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 240, 4, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 480, 4, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 480, 2, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((60,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((120,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 12, 120), torch.float32), ((1, 12, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 8, 12, 12), torch.float32), ((1, 8, 12, 1), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((480,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels",
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 12, 6625), torch.float32), ((1, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((64,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_34_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_densenet_densenet169_img_cls_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((128,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_34_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_densenet_densenet169_img_cls_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((256,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_34_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_densenet_densenet169_img_cls_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_speecht5_tts_tts_text_to_speech_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((512,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_34_img_cls_paddlemodels",
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_densenet_densenet169_img_cls_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pd_resnet_18_img_cls_paddlemodels",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide3,
        [((96,), torch.bfloat16)],
        {
            "model_names": [
                "pt_densenet_densenet161_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_hrnet_hrnet_w48_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_densenet_densenet121_xray_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((192,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_yolox_yolox_m_obj_det_torchhub",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_regnet_regnet_y_040_img_cls_hf",
                    "pt_vovnet_vovnet57_img_cls_osmr",
                    "pt_yolox_yolox_tiny_obj_det_torchhub",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_vovnet_ese_vovnet39b_img_cls_timm",
                    "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_inception_inceptionv4_img_cls_osmr",
                    "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                    "pt_vovnet_vovnet39_img_cls_osmr",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                    "pt_hrnet_hrnet_w48_img_cls_timm",
                    "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                    "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                    "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                    "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                    "pt_yolo_v6_yolov6m_obj_det_torchhub",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_inception_inception_v4_img_cls_timm",
                    "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                    "pt_vovnet_ese_vovnet99b_img_cls_timm",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                    "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((144,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                    "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                    "pt_regnet_regnet_y_064_img_cls_hf",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_img_cls_timm",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                    "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((240,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                    "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_hrnet_hrnet_w30_img_cls_timm",
                    "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                    "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                    "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                    "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((288,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_regnet_regnet_y_064_img_cls_hf",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                    "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((336,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                    "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                    "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                    "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((384,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_yolox_yolox_m_obj_det_torchhub",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_yolox_yolox_tiny_obj_det_torchhub",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_inception_inceptionv4_img_cls_osmr",
                    "pt_hrnet_hrnet_w48_img_cls_timm",
                    "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                    "pt_vovnet_vovnet27s_img_cls_osmr",
                    "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                    "pt_yolo_v6_yolov6m_obj_det_torchhub",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                    "pt_inception_inception_v4_img_cls_timm",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                    "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((432,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                    "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((480,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                    "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                    "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((528,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((576,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                    "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                    "pt_regnet_regnet_y_064_img_cls_hf",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                    "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((624,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((672,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                    "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                    "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                    "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                    "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((720,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((768,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_yolox_yolox_m_obj_det_torchhub",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_vovnet_vovnet57_img_cls_osmr",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_vovnet_ese_vovnet39b_img_cls_timm",
                    "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_vovnet_vovnet39_img_cls_osmr",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                    "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                    "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                    "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                    "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                    "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                    "pt_yolo_v6_yolov6m_obj_det_torchhub",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                    "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                    "pt_vovnet_ese_vovnet99b_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((816,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((864,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((912,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((960,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                    "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                    "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                    "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1008,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1056,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                    "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1104,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1152,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1200,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1248,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1296,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision", "pt_regnet_regnet_y_064_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1344,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1392,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_regnet_regnet_y_320_img_cls_hf",
                    "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1440,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1488,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1536,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_xception_xception41_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                    "pt_xception_xception71_img_cls_timm",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_xception_xception71_tf_in1k_img_cls_timm",
                    "pt_xception_xception65_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1584,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1632,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1680,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1728,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1776,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1824,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1872,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1920,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1968,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2016,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_regnet_regnet_y_080_img_cls_hf",
                    "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2064,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2112,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet161_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2160,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2208,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet161_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide3,
        [((16,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_hrnet_hrnet_w18_small_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_monodle_dla34_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((16,), torch.bfloat16)],
        {"model_names": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide3,
        [((32,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_xception_xception41_img_cls_timm",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr",
                "pt_regnet_regnet_y_040_img_cls_hf",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_unet_base_img_seg_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_hrnet_hrnet_w18_small_img_cls_timm",
                "pt_hrnet_hrnet_w30_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_regnet_regnet_y_320_img_cls_hf",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w32_img_cls_timm",
                "pt_hrnet_hrnet_w40_img_cls_timm",
                "pt_hrnet_hrnet_w44_img_cls_timm",
                "pt_hrnet_hrnet_w48_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_regnet_regnet_y_120_img_cls_hf",
                "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_regnet_regnet_y_064_img_cls_hf",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                "pt_regnet_regnet_y_080_img_cls_hf",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_regnet_regnet_x_3_2gf_img_cls_torchvision",
                "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_regnet_regnet_y_128gf_img_cls_torchvision",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w64_img_cls_timm",
                "pt_regnet_regnet_x_32gf_img_cls_torchvision",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                "pt_xception_xception65_img_cls_timm",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                "pt_yolo_v3_default_obj_det_github",
                "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
                "pt_monodle_dla34_obj_det_github",
                "pt_regnet_regnet_y_160_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((32,), torch.bfloat16)],
        {"model_names": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide3,
        [((64,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_64x4d_osmr_img_cls_osmr",
                "pt_retinanet_retinanet_rn152fpn_obj_det_github",
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_resnext_resnext26_32x4d_osmr_img_cls_osmr",
                "pt_retinanet_retinanet_rn34fpn_obj_det_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_unet_base_img_seg_torchhub",
                "pt_vgg_torchvision_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_glpn_kitti_default_depth_estimation_hf",
                "pt_hrnet_hrnet_w18_small_img_cls_timm",
                "pt_hrnet_hrnet_w30_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_vgg_torchvision_vgg11_bn_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w32_img_cls_timm",
                "pt_hrnet_hrnet_w40_img_cls_timm",
                "pt_hrnet_hrnet_w44_img_cls_timm",
                "pt_hrnet_hrnet_w48_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                "pt_resnet_resnet50_timm_img_cls_timm",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_osmr_img_cls_osmr",
                "pt_vgg_torchvision_vgg19_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_timm_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_timm_img_cls_timm",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_resnext_resnext14_32x4d_osmr_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_torchvision_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_densenet_densenet121_xray_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn101fpn_obj_det_github",
                "pt_retinanet_retinanet_rn18fpn_obj_det_github",
                "pt_retinanet_retinanet_rn50fpn_obj_det_github",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w64_img_cls_timm",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_yolov9_default_obj_det_github",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_yolo_v3_default_obj_det_github",
                "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodle_dla34_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((64,), torch.bfloat16)],
        {"model_names": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide3,
        [((128,), torch.bfloat16)],
        {
            "model_names": [
                "pt_dla_dla46_c_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_retinanet_retinanet_rn152fpn_obj_det_github",
                "pt_unet_carvana_base_img_seg_github",
                "pt_unet_cityscape_img_seg_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_densenet_densenet201_img_cls_torchvision",
                "pt_dla_dla34_visual_bb_torchvision",
                "pt_dla_dla60x_visual_bb_torchvision",
                "pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr",
                "pt_regnet_regnet_y_040_img_cls_hf",
                "pt_resnext_resnext26_32x4d_osmr_img_cls_osmr",
                "pt_retinanet_retinanet_rn34fpn_obj_det_github",
                "pt_vovnet_vovnet57_img_cls_osmr",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_dla_dla102x_visual_bb_torchvision",
                "pt_dla_dla60x_c_visual_bb_torchvision",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_unet_base_img_seg_torchhub",
                "pt_vgg_torchvision_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_timm",
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_small_img_cls_timm",
                "pt_hrnet_hrnet_w30_img_cls_timm",
                "pt_inception_inceptionv4_img_cls_osmr",
                "pt_vgg_torchvision_vgg11_bn_img_cls_torchvision",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w32_img_cls_timm",
                "pt_hrnet_hrnet_w40_img_cls_timm",
                "pt_hrnet_hrnet_w44_img_cls_timm",
                "pt_hrnet_hrnet_w48_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_resnet_resnet50_timm_img_cls_timm",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_dla_dla34_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                "pt_mobilenetv1_basic_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_osmr_img_cls_osmr",
                "pt_vgg_torchvision_vgg19_bn_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_wideresnet_wide_resnet101_2_timm_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_yolox_yolox_nano_obj_det_torchhub",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_wideresnet_wide_resnet50_2_timm_img_cls_timm",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_densenet_densenet169_img_cls_torchvision",
                "pt_dla_dla169_visual_bb_torchvision",
                "pt_dla_dla60_visual_bb_torchvision",
                "pt_hrnet_hrnet_w18_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_resnext_resnext14_32x4d_osmr_img_cls_osmr",
                "pt_ssd300_resnet50_base_img_cls_torchhub",
                "pt_vgg_torchvision_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_densenet_densenet121_xray_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_retinanet_retinanet_rn101fpn_obj_det_github",
                "pt_retinanet_retinanet_rn18fpn_obj_det_github",
                "pt_retinanet_retinanet_rn50fpn_obj_det_github",
                "pt_dla_dla102_visual_bb_torchvision",
                "pt_dla_dla102x2_visual_bb_torchvision",
                "pt_dla_dla46x_c_visual_bb_torchvision",
                "pt_hrnet_hrnet_w64_img_cls_timm",
                "pt_regnet_regnet_x_800mf_img_cls_torchvision",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_yolov9_default_obj_det_github",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_yolo_v3_default_obj_det_github",
                "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodle_dla34_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((128,), torch.bfloat16)],
        {"model_names": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Divide3,
            [((256,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_dla_dla46_c_visual_bb_torchvision",
                    "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                    "pt_resnet_resnet101_img_cls_torchvision",
                    "pt_resnext_resnext101_64x4d_osmr_img_cls_osmr",
                    "pt_retinanet_retinanet_rn152fpn_obj_det_github",
                    "pt_unet_carvana_base_img_seg_github",
                    "pt_unet_cityscape_img_seg_osmr",
                    "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                    "pt_xception_xception41_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_dla_dla34_visual_bb_torchvision",
                    "pt_dla_dla60x_visual_bb_torchvision",
                    "pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr",
                    "pt_resnext_resnext26_32x4d_osmr_img_cls_osmr",
                    "pt_retinanet_retinanet_rn34fpn_obj_det_github",
                    "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                    "pt_vovnet_vovnet57_img_cls_osmr",
                    "pt_yolox_yolox_s_obj_det_torchhub",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_dla_dla102x_visual_bb_torchvision",
                    "pt_dla_dla60x_c_visual_bb_torchvision",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                    "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                    "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                    "pt_unet_base_img_seg_torchhub",
                    "pt_vgg_torchvision_vgg13_bn_img_cls_torchvision",
                    "pt_vgg_vgg19_bn_obj_det_timm",
                    "pt_vovnet_ese_vovnet39b_img_cls_timm",
                    "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                    "pt_yolo_v6_yolov6n_obj_det_torchhub",
                    "pt_fpn_base_img_cls_torchvision",
                    "pt_hrnet_hrnet_w18_small_img_cls_timm",
                    "pt_hrnet_hrnet_w30_img_cls_timm",
                    "pt_inception_inceptionv4_img_cls_osmr",
                    "pt_vgg_torchvision_vgg11_bn_img_cls_torchvision",
                    "pt_vovnet_vovnet39_img_cls_osmr",
                    "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                    "pt_yolox_yolox_darknet_obj_det_torchhub",
                    "pt_yolox_yolox_l_obj_det_torchhub",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                    "pt_hrnet_hrnet_w32_img_cls_timm",
                    "pt_hrnet_hrnet_w40_img_cls_timm",
                    "pt_hrnet_hrnet_w44_img_cls_timm",
                    "pt_hrnet_hrnet_w48_img_cls_timm",
                    "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                    "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                    "pt_resnet_resnet50_timm_img_cls_timm",
                    "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                    "pt_vgg_bn_vgg19b_obj_det_osmr",
                    "pt_vovnet_vovnet27s_img_cls_osmr",
                    "pt_dla_dla34_in1k_img_cls_timm",
                    "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                    "pt_mobilenetv1_basic_img_cls_torchvision",
                    "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                    "pt_resnet_resnet18_img_cls_torchvision",
                    "pt_resnext_resnext50_32x4d_osmr_img_cls_osmr",
                    "pt_vgg_torchvision_vgg19_bn_img_cls_torchvision",
                    "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                    "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                    "pt_wideresnet_wide_resnet101_2_timm_img_cls_timm",
                    "pt_xception_xception71_img_cls_timm",
                    "pt_yolox_yolox_nano_obj_det_torchhub",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                    "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                    "pt_resnet_50_img_cls_hf",
                    "pt_resnet_resnet152_img_cls_torchvision",
                    "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                    "pt_wideresnet_wide_resnet50_2_timm_img_cls_timm",
                    "pt_yolo_v6_yolov6m_obj_det_torchhub",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_dla_dla169_visual_bb_torchvision",
                    "pt_dla_dla60_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w18_img_cls_timm",
                    "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                    "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                    "pt_resnext_resnext14_32x4d_osmr_img_cls_osmr",
                    "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                    "pt_ssd300_resnet50_base_img_cls_torchhub",
                    "pt_vgg_torchvision_vgg16_bn_img_cls_torchvision",
                    "pt_vgg_vgg19_bn_obj_det_torchhub",
                    "pt_xception_xception71_tf_in1k_img_cls_timm",
                    "pt_yolo_v6_yolov6s_obj_det_torchhub",
                    "pt_yolo_world_default_obj_det_github",
                    "pt_yolov10_yolov10n_obj_det_github",
                    "pt_yolov8_yolov8n_obj_det_github",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                    "pt_inception_inception_v4_img_cls_timm",
                    "pt_resnet_resnet34_img_cls_torchvision",
                    "pt_resnet_resnet50_img_cls_torchvision",
                    "pt_retinanet_retinanet_rn101fpn_obj_det_github",
                    "pt_retinanet_retinanet_rn18fpn_obj_det_github",
                    "pt_retinanet_retinanet_rn50fpn_obj_det_github",
                    "pt_dla_dla102_visual_bb_torchvision",
                    "pt_dla_dla102x2_visual_bb_torchvision",
                    "pt_dla_dla46x_c_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w64_img_cls_timm",
                    "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                    "pt_vovnet_ese_vovnet99b_img_cls_timm",
                    "pt_xception_xception65_img_cls_timm",
                    "pt_yolov9_default_obj_det_github",
                    "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                    "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                    "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                    "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                    "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                    "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                    "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                    "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                    "pt_yolo_v3_default_obj_det_github",
                    "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
                    "pt_vgg_bn_vgg19_obj_det_osmr",
                    "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                    "pt_monodle_dla34_obj_det_github",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((256,), torch.bfloat16)],
            {"model_names": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide3,
        [((24,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((40,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((80,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_regnet_regnet_x_8gf_img_cls_torchvision",
                "pt_hrnet_hrnet_w40_img_cls_timm",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((112,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((320,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_yolov8_yolov8x_obj_det_github",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_yolox_yolox_x_obj_det_torchhub",
                    "pt_inception_inceptionv4_img_cls_osmr",
                    "pt_hrnet_hrnet_w40_img_cls_timm",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                    "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                    "pt_yolov10_yolov10x_obj_det_github",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_inception_inception_v4_img_cls_timm",
                    "pt_regnet_regnet_y_800mf_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                    "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1280,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b1_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnet_b0_aa_in1k_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b0_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_yolox_yolox_x_obj_det_torchhub",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                    "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b0_ra_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                    "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                    "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                    "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide3,
        [((48,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_yolox_yolox_tiny_obj_det_torchhub",
                "pt_googlenet_googlenet_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                "pt_hrnet_hrnet_w48_img_cls_timm",
                "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((56,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((160,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_yolov8_yolov8x_obj_det_github",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                    "pt_vovnet_vovnet57_img_cls_osmr",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                    "pt_vovnet_ese_vovnet39b_img_cls_timm",
                    "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                    "pt_yolox_yolox_x_obj_det_torchhub",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_vovnet_vovnet39_img_cls_osmr",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                    "pt_hrnet_hrnet_w40_img_cls_timm",
                    "pt_regnet_regnet_x_400mf_img_cls_torchvision",
                    "pt_ghostnet_ghostnet_100_img_cls_timm",
                    "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                    "pt_mobilenetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf",
                    "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                    "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                    "pt_yolov10_yolov10x_obj_det_github",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                    "pt_vovnet_ese_vovnet99b_img_cls_timm",
                    "pt_mobilenetv2_basic_img_cls_torchhub",
                    "pt_mobilenetv2_mobilenet_v2_torchvision_img_cls_torchvision",
                    "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((272,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((448,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                    "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                    "pt_inception_inceptionv4_img_cls_osmr",
                    "pt_regnet_regnet_y_120_img_cls_hf",
                    "pt_regnet_regnet_y_080_img_cls_hf",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_inception_inception_v4_img_cls_timm",
                    "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                    "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                    "pt_regnet_regnet_y_160_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2688,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1792,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b4_ra2_in1k_img_cls_timm",
                    "pt_efficientnet_timm_efficientnet_b4_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((224,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_vovnet_vovnet57_img_cls_osmr",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                    "pt_vovnet_ese_vovnet39b_img_cls_timm",
                    "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                    "pt_inception_inceptionv4_img_cls_osmr",
                    "pt_vovnet_vovnet39_img_cls_osmr",
                    "pt_regnet_regnet_y_120_img_cls_hf",
                    "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                    "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_inception_inception_v4_img_cls_timm",
                    "pt_regnet_regnet_y_16gf_img_cls_torchvision",
                    "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                    "pt_vovnet_ese_vovnet99b_img_cls_timm",
                    "pt_regnet_regnet_y_160_img_cls_hf",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2304,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((640,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b7_img_cls_torchvision",
                    "pt_yolov8_yolov8x_obj_det_github",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_yolox_yolox_x_obj_det_torchhub",
                    "pt_yolov10_yolov10x_obj_det_github",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((3840,), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2560,), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide3,
        [((18,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((36,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((72,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                "pt_regnet_regnet_x_1_6gf_img_cls_torchvision",
                "pt_regnet_regnet_y_3_2gf_img_cls_torchvision",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_hrnet_hrnet_w18_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((1024,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                    "pt_resnet_resnet101_img_cls_torchvision",
                    "pt_resnext_resnext101_64x4d_osmr_img_cls_osmr",
                    "pt_retinanet_retinanet_rn152fpn_obj_det_github",
                    "pt_unet_carvana_base_img_seg_github",
                    "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                    "pt_xception_xception41_img_cls_timm",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_dla_dla60x_visual_bb_torchvision",
                    "pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr",
                    "pt_resnext_resnext26_32x4d_osmr_img_cls_osmr",
                    "pt_vovnet_vovnet57_img_cls_osmr",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_dla_dla102x_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                    "pt_vovnet_ese_vovnet39b_img_cls_timm",
                    "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                    "pt_hrnet_hrnet_w18_small_img_cls_timm",
                    "pt_hrnet_hrnet_w30_img_cls_timm",
                    "pt_vovnet_vovnet39_img_cls_osmr",
                    "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                    "pt_yolox_yolox_darknet_obj_det_torchhub",
                    "pt_yolox_yolox_l_obj_det_torchhub",
                    "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                    "pt_hrnet_hrnet_w32_img_cls_timm",
                    "pt_hrnet_hrnet_w40_img_cls_timm",
                    "pt_hrnet_hrnet_w44_img_cls_timm",
                    "pt_hrnet_hrnet_w48_img_cls_timm",
                    "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                    "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                    "pt_resnet_resnet50_timm_img_cls_timm",
                    "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                    "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                    "pt_mobilenetv1_basic_img_cls_torchvision",
                    "pt_resnext_resnext50_32x4d_osmr_img_cls_osmr",
                    "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                    "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                    "pt_wideresnet_wide_resnet101_2_timm_img_cls_timm",
                    "pt_xception_xception71_img_cls_timm",
                    "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                    "pt_resnet_50_img_cls_hf",
                    "pt_resnet_resnet152_img_cls_torchvision",
                    "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                    "pt_wideresnet_wide_resnet50_2_timm_img_cls_timm",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_dla_dla169_visual_bb_torchvision",
                    "pt_dla_dla60_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w18_img_cls_timm",
                    "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                    "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                    "pt_resnext_resnext14_32x4d_osmr_img_cls_osmr",
                    "pt_ssd300_resnet50_base_img_cls_torchhub",
                    "pt_xception_xception71_tf_in1k_img_cls_timm",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                    "pt_resnet_resnet50_img_cls_torchvision",
                    "pt_retinanet_retinanet_rn101fpn_obj_det_github",
                    "pt_retinanet_retinanet_rn50fpn_obj_det_github",
                    "pt_dla_dla102_visual_bb_torchvision",
                    "pt_dla_dla102x2_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w64_img_cls_timm",
                    "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                    "pt_vovnet_ese_vovnet99b_img_cls_timm",
                    "pt_xception_xception65_img_cls_timm",
                    "pt_yolo_v3_default_obj_det_github",
                    "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1024,), torch.bfloat16)],
            {"model_names": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((512,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                    "pt_resnet_resnet101_img_cls_torchvision",
                    "pt_resnext_resnext101_64x4d_osmr_img_cls_osmr",
                    "pt_retinanet_retinanet_rn152fpn_obj_det_github",
                    "pt_unet_carvana_base_img_seg_github",
                    "pt_unet_cityscape_img_seg_osmr",
                    "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_dla_dla34_visual_bb_torchvision",
                    "pt_dla_dla60x_visual_bb_torchvision",
                    "pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr",
                    "pt_regnet_regnet_y_040_img_cls_hf",
                    "pt_resnext_resnext26_32x4d_osmr_img_cls_osmr",
                    "pt_retinanet_retinanet_rn34fpn_obj_det_github",
                    "pt_vovnet_vovnet57_img_cls_osmr",
                    "pt_yolox_yolox_s_obj_det_torchhub",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_dla_dla102x_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                    "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                    "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                    "pt_unet_base_img_seg_torchhub",
                    "pt_vgg_torchvision_vgg13_bn_img_cls_torchvision",
                    "pt_vgg_vgg19_bn_obj_det_timm",
                    "pt_vovnet_ese_vovnet39b_img_cls_timm",
                    "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_hrnet_hrnet_w18_small_img_cls_timm",
                    "pt_hrnet_hrnet_w30_img_cls_timm",
                    "pt_inception_inceptionv4_img_cls_osmr",
                    "pt_vgg_torchvision_vgg11_bn_img_cls_torchvision",
                    "pt_vovnet_vovnet39_img_cls_osmr",
                    "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                    "pt_yolox_yolox_darknet_obj_det_torchhub",
                    "pt_yolox_yolox_l_obj_det_torchhub",
                    "pt_efficientnet_hf_hub_timm_tf_efficientnetv2_s_in21k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                    "pt_hrnet_hrnet_w32_img_cls_timm",
                    "pt_hrnet_hrnet_w40_img_cls_timm",
                    "pt_hrnet_hrnet_w44_img_cls_timm",
                    "pt_hrnet_hrnet_w48_img_cls_timm",
                    "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                    "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                    "pt_resnet_resnet50_timm_img_cls_timm",
                    "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                    "pt_vgg_bn_vgg19b_obj_det_osmr",
                    "pt_vovnet_vovnet27s_img_cls_osmr",
                    "pt_dla_dla34_in1k_img_cls_timm",
                    "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                    "pt_mobilenetv1_basic_img_cls_torchvision",
                    "pt_resnet_resnet18_img_cls_torchvision",
                    "pt_resnext_resnext50_32x4d_osmr_img_cls_osmr",
                    "pt_vgg_torchvision_vgg19_bn_img_cls_torchvision",
                    "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                    "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                    "pt_wideresnet_wide_resnet101_2_timm_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnetv2_rw_s_ra2_in1k_img_cls_timm",
                    "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                    "pt_resnet_50_img_cls_hf",
                    "pt_resnet_resnet152_img_cls_torchvision",
                    "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                    "pt_wideresnet_wide_resnet50_2_timm_img_cls_timm",
                    "pt_yolo_v6_yolov6m_obj_det_torchhub",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_dla_dla169_visual_bb_torchvision",
                    "pt_dla_dla60_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w18_img_cls_timm",
                    "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                    "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                    "pt_resnext_resnext14_32x4d_osmr_img_cls_osmr",
                    "pt_ssd300_resnet50_base_img_cls_torchhub",
                    "pt_vgg_torchvision_vgg16_bn_img_cls_torchvision",
                    "pt_vgg_vgg19_bn_obj_det_torchhub",
                    "pt_yolo_v6_yolov6s_obj_det_torchhub",
                    "pt_yolo_world_default_obj_det_github",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                    "pt_inception_inception_v4_img_cls_timm",
                    "pt_resnet_resnet34_img_cls_torchvision",
                    "pt_resnet_resnet50_img_cls_torchvision",
                    "pt_retinanet_retinanet_rn101fpn_obj_det_github",
                    "pt_retinanet_retinanet_rn18fpn_obj_det_github",
                    "pt_retinanet_retinanet_rn50fpn_obj_det_github",
                    "pt_dla_dla102_visual_bb_torchvision",
                    "pt_dla_dla102x2_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w64_img_cls_timm",
                    "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                    "pt_vovnet_ese_vovnet99b_img_cls_timm",
                    "pt_yolov9_default_obj_det_github",
                    "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                    "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                    "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                    "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                    "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                    "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                    "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                    "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                    "pt_yolo_v3_default_obj_det_github",
                    "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
                    "pt_vgg_bn_vgg19_obj_det_osmr",
                    "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                    "pt_monodle_dla34_obj_det_github",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((512,), torch.bfloat16)],
            {"model_names": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2048,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_hrnet_hrnetv2_w18_osmr_img_cls_osmr",
                    "pt_resnet_resnet101_img_cls_torchvision",
                    "pt_resnext_resnext101_64x4d_osmr_img_cls_osmr",
                    "pt_retinanet_retinanet_rn152fpn_obj_det_github",
                    "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                    "pt_xception_xception41_img_cls_timm",
                    "pt_hrnet_hrnetv2_w32_osmr_img_cls_osmr",
                    "pt_resnext_resnext26_32x4d_osmr_img_cls_osmr",
                    "pt_hrnet_hrnet_w18_small_v2_osmr_img_cls_osmr",
                    "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_hrnet_hrnet_w18_small_img_cls_timm",
                    "pt_hrnet_hrnet_w30_img_cls_timm",
                    "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                    "pt_hrnet_hrnet_w18_ms_aug_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_v2_img_cls_timm",
                    "pt_hrnet_hrnet_w32_img_cls_timm",
                    "pt_hrnet_hrnet_w40_img_cls_timm",
                    "pt_hrnet_hrnet_w44_img_cls_timm",
                    "pt_hrnet_hrnet_w48_img_cls_timm",
                    "pt_hrnet_hrnetv2_w48_osmr_img_cls_osmr",
                    "pt_resnet_resnet50_timm_img_cls_timm",
                    "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                    "pt_hrnet_hrnetv2_w40_osmr_img_cls_osmr",
                    "pt_resnext_resnext50_32x4d_osmr_img_cls_osmr",
                    "pt_wideresnet_wide_resnet101_2_timm_img_cls_timm",
                    "pt_xception_xception71_img_cls_timm",
                    "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                    "pt_resnet_50_img_cls_hf",
                    "pt_resnet_resnet152_img_cls_torchvision",
                    "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                    "pt_wideresnet_wide_resnet50_2_timm_img_cls_timm",
                    "pt_hrnet_hrnet_w18_img_cls_timm",
                    "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                    "pt_resnext_resnext14_32x4d_osmr_img_cls_osmr",
                    "pt_xception_xception71_tf_in1k_img_cls_timm",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                    "pt_hrnet_hrnet_w18_small_v1_osmr_img_cls_osmr",
                    "pt_resnet_resnet50_img_cls_torchvision",
                    "pt_retinanet_retinanet_rn101fpn_obj_det_github",
                    "pt_retinanet_retinanet_rn50fpn_obj_det_github",
                    "pt_dla_dla102x2_visual_bb_torchvision",
                    "pt_hrnet_hrnet_w64_img_cls_timm",
                    "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                    "pt_xception_xception65_img_cls_timm",
                    "pt_hrnet_hrnetv2_w64_osmr_img_cls_osmr",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2048,), torch.bfloat16)],
            {"model_names": ["pt_unet_qubvel_img_seg_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide0,
        [((1, 8, 256, 2048), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 8, 2048, 256), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide4,
        [((1, 1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 2, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 5, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 8, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_mit_b5_img_cls_hf",
                "pt_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b4_img_cls_hf",
                "pt_segformer_mit_b2_img_cls_hf",
                "pt_segformer_mit_b0_img_cls_hf",
                "pt_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b3_img_cls_hf",
                "pt_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((728,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_xception_xception41_img_cls_timm",
                    "pt_xception_xception71_img_cls_timm",
                    "pt_xception_xception71_tf_in1k_img_cls_timm",
                    "pt_xception_xception65_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 2, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolov8_yolov8x_obj_det_github",
                "pt_yolov10_yolov10x_obj_det_github",
                "pt_yolo_world_default_obj_det_github",
                "pt_yolov10_yolov10n_obj_det_github",
                "pt_yolov8_yolov8n_obj_det_github",
                "pt_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide0,
        [((1, 13, 1536), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 768, 384), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_b16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b32_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 196, 3072), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_b16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 3, 192, 640), torch.float32)],
        {
            "model_names": [
                "onnx_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "onnx_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "onnx_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "onnx_monodepth2_mono_640x192_depth_prediction_torchvision",
                "onnx_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "onnx_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 384, 3000), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 384, 1500), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1500, 1536), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1, 1536), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 2, 8400), torch.float32)],
        {
            "model_names": [
                "onnx_yolov10_default_obj_det_github",
                "onnx_yolov8_default_obj_det_github",
                "onnx_yolov9_default_obj_det_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 14, 768), torch.float32), ((1, 14, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 12, 14, 14), torch.float32), ((1, 12, 14, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 9, 768), torch.float32), ((1, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 12, 9, 9), torch.float32), ((1, 12, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_qa_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_seq_cls_padlenlp",
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 197, 768), torch.float32), ((1, 197, 1), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 12, 197, 197), torch.float32), ((1, 12, 197, 1), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 768), torch.float32), ((1, 1), torch.float32)],
        {
            "model_names": [
                "pd_clip_vision_openai_clip_vit_base_patch16_img_enc_padlenlp",
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "pd_chineseclip_vision_ofa_sys_chinese_clip_vit_base_patch16_img_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((8,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 8, 16, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((40,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((48,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 48, 8, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 48, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((24,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 120, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 64, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((72,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 72, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((144,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 144, 4, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 144, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((288,), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 288, 2, 50), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 25, 97), torch.float32), ((1, 25, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Divide3,
            [((352,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_hrnet_hrnet_w44_img_cls_timm",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((416,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((544,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((608,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((704,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((736,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((800,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((832,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((896,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                    "pt_regnet_regnet_y_120_img_cls_hf",
                    "pt_regnet_regnet_y_080_img_cls_hf",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                    "pt_regnet_regnet_y_8gf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((928,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((992,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet121_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                    "pt_densenet_densenet121_xray_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1088,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_regnet_regnet_y_040_img_cls_hf",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1120,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1184,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1216,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1312,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1376,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1408,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1472,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1504,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1568,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1600,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1664,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_densenet_densenet201_img_cls_torchvision",
                    "pt_densenet_densenet169_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1696,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1760,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1856,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1888,), torch.bfloat16)],
            {
                "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide3,
        [((88,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w44_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((120,), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_hrnet_hrnet_w30_img_cls_timm",
                "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_regnet_regnet_y_1_6gf_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((208,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                    "pt_googlenet_googlenet_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b2_img_cls_torchvision",
                    "pt_regnet_regnet_y_400mf_img_cls_torchvision",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 16, 112, 112), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 72, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 120, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 240, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 240, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((200,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                    "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_efficientnet_efficientnet_b6_img_cls_torchvision",
                    "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 200, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((184,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                    "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                    "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 184, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 480, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 480, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 672, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 672, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 672, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 960, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 960, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide0,
        [((1, 128, 3072), torch.float32)],
        {
            "model_names": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Divide0, [((1, 128, 768), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (
        Divide0,
        [((1, 257, 3072), torch.float32)],
        {"model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1024, 512), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_l32_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_l16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 49, 4096), torch.float32)],
        {"model_names": ["onnx_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 256, 1280), torch.float32)],
        {"model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 2048, 768), torch.float32)],
        {"model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 16384, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 1024, 640), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Divide1, [((1, 1, 1), torch.float32)], {"model_names": ["onnx_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (Divide1, [((1, 61, 1), torch.float32)], {"model_names": ["onnx_t5_t5_small_text_gen_hf"], "pcc": 0.99}),
    (
        Divide0,
        [((1, 512, 3000), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 512, 1500), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1500, 2048), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1, 2048), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 12, 97), torch.float32), ((1, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1024,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "onnx_densenet_densenet169_img_cls_torchvision",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((2048,), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 16, 5, 5), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide4,
        [((1, 1, 512, 50176), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 8, 512, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 1, 1, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((64, 4, 64, 32), torch.float32), ((64, 4, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((16, 8, 64, 32), torch.float32), ((16, 8, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((4, 16, 64, 32), torch.float32), ((4, 16, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 32, 64, 32), torch.float32), ((1, 32, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 256, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vovnet_ese_vovnet39b_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 5880, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v6_yolov6n_obj_det_torchhub",
                "pt_yolo_v6_yolov6m_obj_det_torchhub",
                "pt_yolo_v6_yolov6s_obj_det_torchhub",
                "pt_yolo_v6_yolov6l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (Divide0, [((1, 128, 4096), torch.float32)], {"model_names": ["onnx_albert_large_v1_mlm_hf"], "pcc": 0.99}),
    (
        Divide1,
        [((96,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((160,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((192,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((224,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((320,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((352,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((384,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((416,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((448,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((544,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((576,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((608,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((640,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((672,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((704,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((736,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((768,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((800,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((832,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((864,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((896,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((928,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((960,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((992,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet121_img_cls_torchvision",
                "pd_densenet_121_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1056,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1088,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1120,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1152,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1184,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1216,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1248,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1280,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1312,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1344,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1376,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1408,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1440,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1472,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1504,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1536,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1568,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1600,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1632,), torch.float32)],
        {
            "model_names": [
                "onnx_densenet_densenet169_img_cls_torchvision",
                "onnx_densenet_densenet161_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide1,
        [((1664,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 49, 3072), torch.float32)],
        {"model_names": ["onnx_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1, 512, 3025), torch.float32)],
        {"model_names": ["onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 512, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 8, 512, 512), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 1, 1, 512), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 1, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 3072, 128), torch.float32)],
        {"model_names": ["onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 12, 201, 201), torch.float32)],
        {"model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 201, 3072), torch.float32)],
        {"model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1536), torch.float32)],
        {"model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 11, 128), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 12, 11, 11), torch.float32), ((1, 12, 11, 1), torch.float32)],
        {
            "model_names": [
                "pd_albert_chinese_tiny_mlm_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 11, 312), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 15, 768), torch.float32), ((1, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 12, 15, 15), torch.float32), ((1, 12, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 8, 768), torch.float32), ((1, 8, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 12, 8, 8), torch.float32), ((1, 12, 8, 1), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_blip_text_salesforce_blip_image_captioning_base_text_enc_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 196, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_l16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 3, 320, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "onnx_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "onnx_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((64, 3, 64, 32), torch.float32), ((64, 3, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 4096, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((16, 6, 64, 32), torch.float32), ((16, 6, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 1024, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((4, 12, 64, 32), torch.float32), ((4, 12, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 256, 1536), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 24, 64, 32), torch.float32), ((1, 24, 64, 32), torch.float32)],
        {
            "model_names": [
                "onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf",
                "pt_swin_swin_v2_s_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 64, 3072), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 577, 768), torch.float32), ((1, 577, 1), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 12, 577, 577), torch.float32), ((1, 12, 577, 1), torch.float32)],
        {
            "model_names": [
                "pd_blip_vision_salesforce_blip_image_captioning_base_img_enc_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((1, 512), torch.float32), ((1, 1), torch.float32)],
        {
            "model_names": [
                "pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp",
                "pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((4, 5, 768), torch.float32), ((4, 5, 1), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((4, 12, 5, 5), torch.float32), ((4, 12, 5, 1), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Divide2,
        [((4, 512), torch.float32), ((4, 1), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 16, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 32, 224, 224), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 48, 112, 112), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 96, 56, 56), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 192, 28, 28), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 384, 14, 14), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 11, 768), torch.float32), ((1, 11, 1), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Divide3,
            [((176,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_hrnet_hrnet_w44_img_cls_timm",
                    "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((304,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((3072,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                    "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 1, 19200, 300), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_default_depth_estimation_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide4,
        [((1, 2, 4800, 300), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_default_depth_estimation_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide4,
        [((1, 5, 1200, 300), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_default_depth_estimation_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide4,
        [((1, 8, 300, 300), torch.bfloat16)],
        {"model_names": ["pt_glpn_kitti_default_depth_estimation_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide3,
        [((30,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w30_img_cls_timm", "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((60,), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_hrnet_hrnetv2_w30_osmr_img_cls_osmr",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((8,), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((168,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision", "pt_regnet_regnet_y_080_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((408,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((232,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_regnet_regnet_y_320_img_cls_hf",
                    "pt_regnet_regnet_y_32gf_img_cls_torchvision",
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((696,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_320_img_cls_hf", "pt_regnet_regnet_y_32gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((3712,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_320_img_cls_hf", "pt_regnet_regnet_y_32gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((216,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((1512,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide0,
        [((1, 197, 3072), torch.float32)],
        {
            "model_names": [
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 197, 1536), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((32,), torch.float32)],
        {
            "model_names": ["pd_mobilenetv1_basic_img_cls_paddlemodels", "pd_mobilenetv2_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
        },
    ),
    (
        Divide3,
        [((44,), torch.bfloat16)],
        {
            "model_names": ["pt_hrnet_hrnet_w44_img_cls_timm", "pt_hrnet_hrnetv2_w44_osmr_img_cls_osmr"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((400,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2240,), torch.bfloat16)],
            {"model_names": ["pt_regnet_regnet_y_120_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 12, 201, 201), torch.bfloat16)],
        {"model_names": ["pt_vilt_vqa_qa_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (Divide0, [((1, 1, 1, 2048), torch.float32)], {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99}),
    (
        Divide3,
        [((12,), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((20,), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((100,), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((92,), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide3,
        [((104,), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((440,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide0,
        [((1, 197, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((240,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((336,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((432,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((528,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((624,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((720,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((816,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((912,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1008,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1104,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1200,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1296,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1392,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1488,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1584,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1680,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1728,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1776,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1824,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1872,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1920,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((1968,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((2016,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((2064,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((2112,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((2160,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide1,
        [((2208,), torch.float32)],
        {"model_names": ["onnx_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 25, 6625), torch.float32), ((1, 25, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Divide3,
            [((136,), torch.bfloat16)],
            {
                "model_names": [
                    "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                    "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                ],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 24, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 12, 204, 204), torch.bfloat16)],
        {"model_names": ["pt_vilt_mlm_mlm_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide0,
        [((1, 49, 2048), torch.float32)],
        {"model_names": ["onnx_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 768, 3000), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 768, 1500), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1500, 3072), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1, 3072), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 10, 768), torch.float32), ((1, 10, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 12, 10, 10), torch.float32), ((1, 12, 10, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 16, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 96, 28, 28), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 96, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 96, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 240, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 120, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 144, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 144, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 288, 14, 14), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 288, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 288, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 576, 7, 7), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 576, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((888,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 4, 40, 40), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Divide4,
        [((1, 2, 80, 80), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Divide2,
            [((1, 512, 80, 80), torch.bfloat16), ((1, 512, 80, 80), torch.bfloat16)],
            {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 8, 80, 27), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Divide2,
            [((1, 512, 40, 40), torch.bfloat16), ((1, 512, 40, 40), torch.bfloat16)],
            {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide4,
        [((1, 8, 20, 20), torch.bfloat16)],
        {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    pytest.param(
        (
            Divide2,
            [((1, 512, 20, 20), torch.bfloat16), ((1, 512, 20, 20), torch.bfloat16)],
            {"model_names": ["pt_yolo_world_default_obj_det_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (Divide0, [((1, 128, 8192), torch.float32)], {"model_names": ["onnx_albert_xlarge_v1_mlm_hf"], "pcc": 0.99}),
    (
        Divide0,
        [((1, 6, 3072), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Divide0,
        [((1, 1, 512, 50176), torch.float32)],
        {"model_names": ["onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((2, 4, 768), torch.float32), ((2, 4, 1), torch.float32)],
        {"model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((2, 12, 4, 4), torch.float32), ((2, 12, 4, 1), torch.float32)],
        {"model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((2, 512), torch.float32), ((2, 1), torch.float32)],
        {"model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"], "pcc": 0.99},
    ),
    (
        Divide4,
        [((1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 1, 512, 3025), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide3,
            [((1232,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_16gf_img_cls_torchvision", "pt_regnet_regnet_y_160_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((3024,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_16gf_img_cls_torchvision", "pt_regnet_regnet_y_160_img_cls_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((784,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide0,
        [((1, 64, 334, 334), torch.float32)],
        {"model_names": ["onnx_fuyu_adept_fuyu_8b_clm_hf", "pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1, 19200, 300), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 19200, 256), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 2, 4800, 300), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 4800, 512), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 5, 1200, 300), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 1200, 1280), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 8, 300, 300), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 300, 2048), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Divide3,
            [((2904,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((7392,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((344,), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((3456,), torch.bfloat16)],
            {
                "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide3,
            [((2520,), torch.bfloat16)],
            {
                "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide2,
            [((2, 32, 10, 16384), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
            {
                "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide2,
        [((2, 32, 10, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2, 32, 20, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2, 32, 20, 1024), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2, 32, 40, 1024), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2, 32, 80, 1024), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2, 32, 60, 1024), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((2, 32, 60, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide2,
            [((2, 32, 40, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
            {
                "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide2,
            [((2, 32, 30, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
            {
                "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide2,
            [((2, 32, 30, 16384), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
            {
                "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide2,
        [((2, 32, 20, 16384), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide4,
        [((1, 3, 192, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((100, 8, 9240), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((100, 8, 4480), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((100, 8, 8640), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((100, 8, 17280), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((100, 8, 34240), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Divide2,
        [((1, 25, 34), torch.bfloat16), ((1, 1, 34), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide5,
        [((1, 25, 34, 1), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Divide2,
        [((1, 25, 34), torch.bfloat16), ((1, 25, 1), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide2,
            [((100, 8, 33, 850), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
            {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide2,
            [((100, 8, 16, 850), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
            {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide2,
            [((100, 8, 8, 3350), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
            {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide2,
            [((100, 8, 4, 13400), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
            {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    pytest.param(
        (
            Divide2,
            [((100, 8, 2, 53400), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
            {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide6,
        [((3, 480, 640), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    pytest.param(
        (
            Divide2,
            [((1, 512, 38, 38), torch.bfloat16), ((1, 512, 38, 38), torch.bfloat16)],
            {
                "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Divide0,
        [((1, 384, 4096), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Divide0,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Divide0, [((1, 1, 256), torch.float32)], {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99}),
    (Divide1, [((80,), torch.float32)], {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99}),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Divide")

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
