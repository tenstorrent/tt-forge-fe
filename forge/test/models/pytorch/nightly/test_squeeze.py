# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import forge.op
from forge import ForgeModule

from loguru import logger
import torch

from forge import Tensor, compile
from forge.verify.compare import compare_with_golden
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
import pytest


class Squeeze0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-1)
        return squeeze_output_1


class Squeeze1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-2)
        return squeeze_output_1


class Squeeze2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-4)
        return squeeze_output_1


class Squeeze3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-3)
        return squeeze_output_1


class Squeeze4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=1)
        return squeeze_output_1


def ids_func(param):
    forge_module, shapes_dtypes, _ = param
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Squeeze0,
        [((1, 128, 768, 1), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v2_masked_lm",
                "pt_albert_base_v1_masked_lm",
                "pt_albert_base_v1_token_cls",
                "pt_albert_base_v2_token_cls",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 128, 2048, 1), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_token_cls",
                "pt_albert_xlarge_v2_token_cls",
                "pt_albert_xlarge_v1_masked_lm",
                "pt_albert_xlarge_v2_masked_lm",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 128, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v1_masked_lm",
                "pt_albert_large_v2_token_cls",
                "pt_albert_large_v2_masked_lm",
                "pt_albert_large_v1_token_cls",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 128, 4096, 1), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v1_token_cls",
                "pt_albert_xxlarge_v2_masked_lm",
                "pt_albert_xxlarge_v1_masked_lm",
                "pt_albert_xxlarge_v2_token_cls",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (Squeeze0, [((1, 384, 1), torch.float32)], {"model_name": ["pt_bert_qa", "pt_distilbert_question_answering"]}),
    (Squeeze0, [((1, 256, 16, 32, 1), torch.float32)], {"model_name": ["pt_codegen_350M_mono"]}),
    (
        Squeeze0,
        [((1, 128, 1), torch.float32)],
        {"model_name": ["pt_dpr_reader_single_nq_base", "pt_dpr_reader_multiset_base"]},
    ),
    (
        Squeeze0,
        [((1, 1), torch.int32)],
        {"model_name": ["pt_opt_125m_seq_cls", "pt_opt_1_3b_seq_cls", "pt_opt_350m_seq_cls"]},
    ),
    (Squeeze0, [((1, 32, 1), torch.float32)], {"model_name": ["pt_opt_1_3b_qa", "pt_opt_350m_qa", "pt_opt_125m_qa"]}),
    (Squeeze1, [((1, 768, 1, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (Squeeze1, [((1, 3072, 1, 128), torch.float32)], {"model_name": ["pt_squeezebert"]}),
    (
        Squeeze2,
        [((1, 1, 1024, 1), torch.float32)],
        {"model_name": ["nbeats_generic", "nbeats_trend", "nbeats_seasonality"]},
    ),
    (
        Squeeze3,
        [((1, 1024, 1), torch.float32)],
        {"model_name": ["nbeats_generic", "nbeats_trend", "nbeats_seasonality"]},
    ),
    (
        Squeeze0,
        [((1, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet121",
                "pt_googlenet",
                "pt_mobilenet_v1_224",
                "pt_mobilenetv3_small_100",
                "pt_ese_vovnet19b_dw",
                "pt_ese_vovnet39b",
                "pt_ese_vovnet99b",
            ]
        },
    ),
    (
        Squeeze2,
        [((1, 1, 1024, 72), torch.float32)],
        {"model_name": ["nbeats_generic", "nbeats_trend", "nbeats_seasonality"]},
    ),
    (
        Squeeze3,
        [((1, 1024, 72), torch.float32)],
        {"model_name": ["nbeats_generic", "nbeats_trend", "nbeats_seasonality"]},
    ),
    (Squeeze1, [((1, 9216, 1, 1), torch.float32)], {"model_name": ["pt_alexnet_torchhub", "pt_rcnn"]}),
    (Squeeze0, [((1, 9216, 1), torch.float32)], {"model_name": ["pt_alexnet_torchhub", "pt_rcnn"]}),
    (Squeeze0, [((1, 384, 196, 1), torch.float32)], {"model_name": ["pt_deit_small_patch16_224"]}),
    (
        Squeeze0,
        [((1, 768, 196, 1), torch.float32)],
        {
            "model_name": [
                "pt_deit_base_patch16_224",
                "pt_deit_base_distilled_patch16_224",
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
                "pt_vit_base_patch16_224",
            ]
        },
    ),
    (Squeeze0, [((1, 192, 196, 1), torch.float32)], {"model_name": ["pt_deit_tiny_patch16_224"]}),
    (
        Squeeze1,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_densenet121",
                "pt_googlenet",
                "pt_mobilenet_v1_224",
                "pt_mobilenetv3_small_100",
                "pt_ese_vovnet19b_dw",
                "pt_ese_vovnet39b",
                "pt_ese_vovnet99b",
            ]
        },
    ),
    (Squeeze1, [((1, 2208, 1, 1), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (Squeeze0, [((1, 2208, 1), torch.float32)], {"model_name": ["pt_densenet_161"]}),
    (Squeeze1, [((1, 1664, 1, 1), torch.float32)], {"model_name": ["pt_densenet_169"]}),
    (Squeeze0, [((1, 1664, 1), torch.float32)], {"model_name": ["pt_densenet_169"]}),
    (Squeeze1, [((1, 1920, 1, 1), torch.float32)], {"model_name": ["pt_densenet_201"]}),
    (Squeeze0, [((1, 1920, 1), torch.float32)], {"model_name": ["pt_densenet_201"]}),
    (
        Squeeze1,
        [((1, 1792, 1, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Squeeze0,
        [((1, 1792, 1), torch.float32)],
        {"model_name": ["pt_efficientnet_b4_timm", "pt_efficientnet_b4_torchvision"]},
    ),
    (
        Squeeze1,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_ghostnet_100",
                "mobilenetv2_basic",
                "mobilenetv2_160",
                "mobilenetv2_96",
                "mobilenetv2_timm",
                "mobilenetv2_224",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 1280, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_b0_torchvision",
                "pt_efficientnet_b0_timm",
                "pt_ghostnet_100",
                "mobilenetv2_basic",
                "mobilenetv2_160",
                "mobilenetv2_96",
                "mobilenetv2_timm",
                "mobilenetv2_224",
                "pt_mobilenetv3_large_100",
            ]
        },
    ),
    (
        Squeeze1,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_timm_hrnet_w30",
                "pt_hrnet_timm_hrnet_w32",
                "pt_hrnet_timm_hrnet_w48",
                "pt_hrnet_timm_hrnet_w40",
                "pt_hrnet_timm_hrnet_w44",
                "pt_hrnet_timm_hrnet_w18_small",
                "pt_hrnet_timm_hrnet_w64",
                "pt_hrnet_timm_hrnet_w18_small_v2",
                "pt_resnet50_timm",
                "pt_resnet50",
                "pt_resnext50_torchhub",
                "pt_resnext101_torchhub",
                "pt_resnext101_fb_wsl",
                "pt_wide_resnet101_2_timm",
                "pt_wide_resnet50_2_hub",
                "pt_wide_resnet50_2_timm",
                "pt_wide_resnet101_2_hub",
                "pt_xception65_timm",
                "pt_xception71_timm",
                "pt_xception41_timm",
                "pt_xception_timm",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 2048, 1), torch.float32)],
        {
            "model_name": [
                "pt_hrnet_timm_hrnet_w18",
                "pt_hrnet_timm_hrnet_w30",
                "pt_hrnet_timm_hrnet_w32",
                "pt_hrnet_timm_hrnet_w48",
                "pt_hrnet_timm_hrnet_w40",
                "pt_hrnet_timm_hrnet_w44",
                "pt_hrnet_timm_hrnet_w18_small",
                "pt_hrnet_timm_hrnet_w64",
                "pt_hrnet_timm_hrnet_w18_small_v2",
                "pt_resnet50_timm",
                "pt_resnet50",
                "pt_resnext50_torchhub",
                "pt_resnext101_torchhub",
                "pt_resnext101_fb_wsl",
                "pt_wide_resnet101_2_timm",
                "pt_wide_resnet50_2_hub",
                "pt_wide_resnet50_2_timm",
                "pt_wide_resnet101_2_hub",
                "pt_xception65_timm",
                "pt_xception71_timm",
                "pt_xception41_timm",
                "pt_xception_timm",
            ]
        },
    ),
    (Squeeze1, [((1, 1536, 1, 1), torch.float32)], {"model_name": ["pt_timm_inception_v4"]}),
    (Squeeze0, [((1, 1536, 1), torch.float32)], {"model_name": ["pt_timm_inception_v4"]}),
    (Squeeze0, [((1, 1024, 49, 1), torch.float32)], {"model_name": ["pt_mixer_l32_224"]}),
    (
        Squeeze4,
        [((1, 1, 1024), torch.float32)],
        {"model_name": ["pt_mixer_l32_224", "pt_mixer_l16_224", "pt_mixer_l16_224_in21k"]},
    ),
    (
        Squeeze0,
        [((1, 1024, 196, 1), torch.float32)],
        {"model_name": ["pt_mixer_l16_224", "pt_mixer_l16_224_in21k", "pt_vit_large_patch16_224"]},
    ),
    (Squeeze0, [((1, 512, 49, 1), torch.float32)], {"model_name": ["pt_mixer_s32_224"]}),
    (
        Squeeze4,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_mixer_s32_224",
                "pt_mixer_s16_224",
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_mit_b2",
                "pt_mit_b3",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze4,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": [
                "pt_mixer_b16_224_miil_in21k",
                "pt_mixer_b16_224",
                "pt_mixer_b32_224",
                "pt_mixer_b16_224_in21k",
                "pt_mixer_b16_224_miil",
            ]
        },
    ),
    (Squeeze0, [((1, 512, 196, 1), torch.float32)], {"model_name": ["pt_mixer_s16_224"]}),
    (Squeeze0, [((1, 768, 49, 1), torch.float32)], {"model_name": ["pt_mixer_b32_224"]}),
    (Squeeze1, [((1, 768, 1, 1), torch.float32)], {"model_name": ["pt_mobilenet_v1_192"]}),
    (Squeeze0, [((1, 768, 1), torch.float32)], {"model_name": ["pt_mobilenet_v1_192"]}),
    (Squeeze1, [((1, 576, 1, 1), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (Squeeze0, [((1, 576, 1), torch.float32)], {"model_name": ["pt_mobilenet_v3_small"]}),
    (Squeeze1, [((1, 960, 1, 1), torch.float32)], {"model_name": ["pt_mobilenet_v3_large"]}),
    (Squeeze0, [((1, 960, 1), torch.float32)], {"model_name": ["pt_mobilenet_v3_large"]}),
    (
        Squeeze1,
        [((1, 1, 1, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Squeeze1, [((1, 512, 1, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (Squeeze1, [((1, 50176, 1, 512), torch.float32)], {"model_name": ["pt_vision_perceiver_learned"]}),
    (
        Squeeze1,
        [((1, 512, 1, 1024), torch.float32)],
        {"model_name": ["pt_vision_perceiver_learned", "pt_vision_perceiver_conv", "pt_vision_perceiver_fourier"]},
    ),
    (Squeeze1, [((1, 512, 1, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Squeeze1, [((1, 3025, 1, 322), torch.float32)], {"model_name": ["pt_vision_perceiver_conv"]}),
    (Squeeze1, [((1, 512, 1, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Squeeze1, [((1, 50176, 1, 261), torch.float32)], {"model_name": ["pt_vision_perceiver_fourier"]}),
    (Squeeze1, [((1, 1088, 1, 1), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (Squeeze0, [((1, 1088, 1), torch.float32)], {"model_name": ["pt_regnet_y_040"]}),
    (
        Squeeze0,
        [((1, 64, 16384, 1), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze1,
        [((1, 16384, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze1,
        [((1, 256, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 256, 16384, 1), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 512, 4096, 1), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 320, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 1280, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 512, 256, 1), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 2048, 256, 1), torch.float32)],
        {
            "model_name": [
                "pt_mit_b4",
                "pt_mit_b1",
                "pt_segformer_b4_finetuned_ade_512_512",
                "pt_mit_b2",
                "pt_segformer_b1_finetuned_ade_512_512",
                "pt_mit_b3",
                "pt_segformer_b2_finetuned_ade_512_512",
                "pt_segformer_b3_finetuned_ade_512_512",
                "pt_mit_b5",
            ]
        },
    ),
    (
        Squeeze0,
        [((1, 32, 16384, 1), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze1,
        [((1, 16384, 1, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze1,
        [((1, 256, 1, 32), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze0,
        [((1, 128, 16384, 1), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze0,
        [((1, 64, 4096, 1), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze0,
        [((1, 256, 4096, 1), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze0,
        [((1, 160, 1024, 1), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze0,
        [((1, 640, 1024, 1), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze0,
        [((1, 256, 256, 1), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (
        Squeeze0,
        [((1, 1024, 256, 1), torch.float32)],
        {"model_name": ["pt_segformer_b0_finetuned_ade_512_512", "pt_mit_b0"]},
    ),
    (Squeeze4, [((1, 1, 256), torch.float32)], {"model_name": ["pt_mit_b0"]}),
    (Squeeze0, [((1, 96, 4096, 1), torch.float32)], {"model_name": ["pt_swinv2_tiny_patch4_window8_256"]}),
    (Squeeze1, [((1, 25088, 1, 1), torch.float32)], {"model_name": ["pt_vgg_19_hf", "pt_vgg_bn19_torchhub"]}),
    (Squeeze0, [((1, 25088, 1), torch.float32)], {"model_name": ["pt_vgg_19_hf", "pt_vgg_bn19_torchhub"]}),
    (Squeeze1, [((1, 4096, 1, 1), torch.float32)], {"model_name": ["pt_vgg19_bn_timm"]}),
    (Squeeze0, [((1, 4096, 1), torch.float32)], {"model_name": ["pt_vgg19_bn_timm"]}),
    (
        Squeeze0,
        [((1, 85, 6400, 1), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Squeeze0,
        [((1, 85, 1600, 1), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (
        Squeeze0,
        [((1, 85, 400, 1), torch.float32)],
        {"model_name": ["pt_yolox_m", "pt_yolox_s", "pt_yolox_darknet", "pt_yolox_x", "pt_yolox_l"]},
    ),
    (Squeeze0, [((1, 85, 2704, 1), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Squeeze0, [((1, 85, 676, 1), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
    (Squeeze0, [((1, 85, 169, 1), torch.float32)], {"model_name": ["pt_yolox_nano", "pt_yolox_tiny"]}),
]


@pytest.mark.push
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, record_property):
    record_property("frontend", "tt-forge-fe")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    for metadata_name, metadata_value in metadata.items():
        record_property(metadata_name, metadata_value)

    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype) for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)
    framework_model.process_framework_parameters()

    compiled_model = compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
