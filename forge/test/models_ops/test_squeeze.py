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
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=3)
        return squeeze_output_1


class Squeeze2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-2)
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
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=2)
        return squeeze_output_1


class Squeeze5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-4)
        return squeeze_output_1


class Squeeze6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=1)
        return squeeze_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Squeeze0,
        [((1, 384, 1), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze1,
        [((1, 100, 8, 1, 280), torch.float32)],
        {
            "model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "3"},
        },
    ),
    (
        Squeeze2,
        [((1, 9216, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_alexnet_base_img_cls_paddlemodels",
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_mnist_base_img_cls_github",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 9216, 1), torch.float32)],
        {
            "model_name": [
                "pd_alexnet_base_img_cls_paddlemodels",
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_mnist_base_img_cls_github",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_googlenet_base_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_googlenet_base_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-3"},
        },
    ),
    (
        Squeeze2,
        [((1, 1152, 1, 1), torch.float32)],
        {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 1152, 1), torch.float32)],
        {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 1280, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 1280, 1), torch.float32)],
        {
            "model_name": [
                "pd_mobilenetv2_basic_img_cls_paddlemodels",
                "pt_efficientnet_efficientnet_b0_img_cls_timm",
                "pt_efficientnet_efficientnet_b0_img_cls_torchvision",
                "pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_ghostnet_ghostnet_100_img_cls_timm",
                "pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm",
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 512, 1), torch.float32)],
        {
            "model_name": [
                "pd_resnet_18_img_cls_paddlemodels",
                "pd_resnet_34_img_cls_paddlemodels",
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze4,
        [((1, 512, 1), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "op_params": {"dim": "2"}},
    ),
    (
        Squeeze2,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_name": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 2048, 1), torch.float32)],
        {
            "model_name": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnet_w32_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet50_img_cls_torchvision",
                "pt_regnet_regnet_x_16gf_img_cls_torchvision",
                "pt_resnet_resnet101_img_cls_torchvision",
                "pt_resnet_resnet50_img_cls_torchvision",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_hf",
                "pt_resnet_50_img_cls_timm",
                "pt_resnext_resnext50_32x4d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub",
                "pt_wideresnet_wide_resnet50_2_img_cls_timm",
                "pt_wideresnet_wide_resnet50_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_torchvision",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_xception_xception71_tf_in1k_img_cls_timm",
                "pt_xception_xception65_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1024, 576, 1), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 128, 2048, 1), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 128, 768, 1), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 128, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 14, 768, 1), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 14, 1), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 9, 768, 1), torch.float32)],
        {
            "model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 128, 4096, 1), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 16, 32, 1), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 128, 1), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1), torch.int32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": [
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 512, 1, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 50176, 1, 512), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 512, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 512, 1, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 3025, 1, 322), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 512, 1, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 50176, 1, 261), torch.float32)],
        {
            "model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 768, 1, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 3072, 1, 128), torch.float32)],
        {
            "model_name": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze5,
        [((1, 1, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-4"},
        },
    ),
    (
        Squeeze5,
        [((1, 1, 1024, 72), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-4"},
        },
    ),
    (
        Squeeze3,
        [((1, 1024, 72), torch.float32)],
        {
            "model_name": [
                "pt_nbeats_generic_basis_clm_hf",
                "pt_nbeats_trend_basis_clm_hf",
                "pt_nbeats_seasionality_basis_clm_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-3"},
        },
    ),
    (
        Squeeze5,
        [((1, 1, 96, 54, 54), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"dim": "-4"}},
    ),
    (
        Squeeze5,
        [((1, 1, 256, 27, 27), torch.float32)],
        {"model_name": ["pt_alexnet_base_img_cls_osmr"], "pcc": 0.99, "op_params": {"dim": "-4"}},
    ),
    (
        Squeeze0,
        [((1, 768, 196, 1), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze6,
        [((1, 1, 768), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1024, 196, 1), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze6,
        [((1, 1, 1024), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "1"},
        },
    ),
    (
        Squeeze0,
        [((1, 384, 196, 1), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 192, 196, 1), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 2208, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 2208, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 1920, 1, 1), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet201_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 1920, 1), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet201_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 1664, 1, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 1664, 1), torch.float32)],
        {"model_name": ["pt_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 1000, 1, 1), torch.float32)],
        {"model_name": ["pt_dla_dla34_in1k_img_cls_timm"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 1000, 1), torch.float32)],
        {"model_name": ["pt_dla_dla34_in1k_img_cls_timm"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 1792, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 1792, 1), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 64, 19200, 1), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 19200, 1, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 300, 1, 64), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 19200, 1), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 128, 4800, 1), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 512, 4800, 1), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 320, 1200, 1), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1280, 1200, 1), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 512, 300, 1), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 2048, 300, 1), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 1, 480, 640), torch.float32)],
        {
            "model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-3"},
        },
    ),
    (
        Squeeze2,
        [((1, 1536, 1, 1), torch.float32)],
        {
            "model_name": ["pt_inception_inception_v4_img_cls_timm", "pt_inception_inception_v4_tf_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 1536, 1), torch.float32)],
        {
            "model_name": ["pt_inception_inception_v4_img_cls_timm", "pt_inception_inception_v4_tf_in1k_img_cls_timm"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 768, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 768, 1), torch.float32)],
        {
            "model_name": [
                "pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 576, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 576, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 960, 1, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 960, 1), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 1088, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 1088, 1), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 7392, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 7392, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 888, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 888, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 3712, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 3712, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 440, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 440, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 2520, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 2520, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 1008, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 1008, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 912, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 912, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 672, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 672, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 2016, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 2016, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 784, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 784, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 1512, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 1512, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 400, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 400, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze2,
        [((1, 3024, 1, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 3024, 1), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 64, 16384, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 16384, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 256, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 16384, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 512, 4096, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 320, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1280, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 512, 256, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 2048, 256, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 32, 16384, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 16384, 1, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 256, 1, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 128, 16384, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 64, 4096, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 4096, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 160, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 640, 1024, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 256, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1024, 256, 1), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze6,
        [((1, 1, 256), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "op_params": {"dim": "1"}},
    ),
    (
        Squeeze6,
        [((1, 1, 512), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "1"},
        },
    ),
    (
        Squeeze0,
        [((1, 96, 3136, 1), torch.float32)],
        {
            "model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 25088, 1, 1), torch.float32)],
        {
            "model_name": [
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 25088, 1), torch.float32)],
        {
            "model_name": [
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze2,
        [((1, 4096, 1, 1), torch.float32)],
        {"model_name": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99, "op_params": {"dim": "-2"}},
    ),
    (
        Squeeze0,
        [((1, 4096, 1), torch.float32)],
        {"model_name": ["pt_vgg_vgg19_bn_obj_det_timm"], "pcc": 0.99, "op_params": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 85, 6400, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 85, 1600, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 85, 400, 1), torch.float32)],
        {
            "model_name": [
                "pt_yolox_yolox_darknet_obj_det_torchhub",
                "pt_yolox_yolox_x_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_s_obj_det_torchhub",
                "pt_yolox_yolox_l_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 85, 2704, 1), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 85, 676, 1), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 85, 169, 1), torch.float32)],
        {
            "model_name": ["pt_yolox_yolox_tiny_obj_det_torchhub", "pt_yolox_yolox_nano_obj_det_torchhub"],
            "pcc": 0.99,
            "op_params": {"dim": "-1"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):

    forge_property_recorder.enable_single_op_details_recording()
    forge_property_recorder.record_forge_op_name("Squeeze")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_name":
            forge_property_recorder.record_op_model_names(metadata_value)
        elif metadata_name == "op_params":
            forge_property_recorder.record_forge_op_args(metadata_value)
        else:
            logger.warning("no utility function in forge property handler")

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
