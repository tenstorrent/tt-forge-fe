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


class Squeeze0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-2)
        return squeeze_output_1


class Squeeze1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-1)
        return squeeze_output_1


class Squeeze2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=2)
        return squeeze_output_1


class Squeeze3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=1)
        return squeeze_output_1


class Squeeze4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-3)
        return squeeze_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Squeeze0,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 2048, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 128, 1), torch.float32)],
        {"model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 1536, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1536, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1792, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1792, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 2304, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 2304, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 2048, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 2048, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 512, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "2"},
        },
    ),
    (
        Squeeze1,
        [((1, 512, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_ssd_resnet34_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 512, 196, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 1, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1024, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv3_ssd_resnet34_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 7392, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 7392, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 2016, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 2016, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 32, 16384, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 16384, 1, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 1, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 128, 16384, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 64, 4096, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 256, 4096, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 160, 1024, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 640, 1024, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 256, 256, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 1024, 256, 1), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 1, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "1"},
        },
    ),
    (
        Squeeze0,
        [((1, 768, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 768, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 25088, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 25088, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 768, 196, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((3, 50, 1, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 85, 6400, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 85, 1600, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 85, 400, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 16384, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 1, 768), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dim": "-2"}},
    ),
    (
        Squeeze1,
        [((1, 1024, 196, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 1, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "1"},
        },
    ),
    (
        Squeeze0,
        [((1, 2560, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 2560, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1280, 1, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1280, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 1, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "1"},
        },
    ),
    (
        Squeeze1,
        [((1, 512, 49, 1), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 3024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 3024, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 3712, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 3712, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 2520, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 2520, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1008, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1008, 1), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 768, 1, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-2"}},
    ),
    (
        Squeeze1,
        [((1, 768, 1), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze1,
        [((1, 192, 196, 1), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 64, 16384, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 16384, 1, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 1, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 256, 16384, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 128, 4096, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 512, 4096, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 320, 1024, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 1280, 1024, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 512, 256, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 2048, 256, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 768, 1, 128), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 3072, 1, 128), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((3, 197, 1, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((3, 197, 1, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Squeeze3,
        [((1, 1, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Squeeze0,
        [((1, 9216, 1, 1), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "-2"}},
    ),
    (
        Squeeze1,
        [((1, 9216, 1), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 288, 1, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze4,
        [((1, 1, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3"},
        },
    ),
    (
        Squeeze4,
        [((1, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-3"},
        },
    ),
    (
        Squeeze1,
        [((1, 256, 16, 32, 1), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 1920, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1920, 1), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1408, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1408, 1), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Squeeze")

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
