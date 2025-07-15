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


class Transpose0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-3, dim1=-2)
        return transpose_output_1


class Transpose1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-2, dim1=-1)
        return transpose_output_1


class Transpose2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-5, dim1=-4)
        return transpose_output_1


class Transpose3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-4, dim1=-2)
        return transpose_output_1


class Transpose4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-3, dim1=-1)
        return transpose_output_1


class Transpose5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-4, dim1=-3)
        return transpose_output_1


class Transpose6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-5, dim1=-3)
        return transpose_output_1


class Transpose7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-5, dim1=-2)
        return transpose_output_1


class Transpose8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, transpose_input_0):
        transpose_output_1 = forge.op.Transpose("", transpose_input_0, dim0=-4, dim1=-1)
        return transpose_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Transpose0,
        [((1, 11, 12, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 26, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((21128, 128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1024, 128), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1024, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 16, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 1024), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((768, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_t5_t5_base_text_gen_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_gpt_gpt2_text_gen_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 128, 12, 64), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 128, 64), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3072, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_t5_t5_base_text_gen_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 3072), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_swin_swin_v2_t_img_cls_torchvision",
                "pt_t5_t5_base_text_gen_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768), torch.float32)],
        {
            "model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1536), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1792), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2304), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_resnet_resnet152_img_cls_torchvision",
                "pt_xception_xception41_img_cls_timm",
                "pt_xception_xception71_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_mobilenetv3_ssd_resnet101_img_cls_torchvision",
                "pt_resnext_resnext101_32x8d_img_cls_torchhub",
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_wideresnet_wide_resnet101_2_img_cls_timm",
                "pt_efficientnet_efficientnet_b5_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet152_img_cls_torchvision",
                "pt_resnet_50_img_cls_timm",
                "pt_xception_xception_img_cls_timm",
                "pt_efficientnet_hf_hub_timm_efficientnet_b5_in12k_ft_in1k_img_cls_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 2048), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 32, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 4), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 2048), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 8, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8192, 2048), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 8192), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 2048), torch.float32)],
        {
            "model_names": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose2,
        [((1, 3, 16, 16, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-4"},
        },
    ),
    (
        Transpose3,
        [((1, 16, 3, 16, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 16, 16, 3, 16), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_base_img_cls_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_vovnet_vovnet27s_img_cls_osmr",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((196, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_s16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_s32_224_img_cls_timm",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1001, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_mobilenetv1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_vovnet_ese_vovnet99b_obj_det_torchhub",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vovnet_vovnet39_img_cls_osmr",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mobilenetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 32, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 16, 256), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 256, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 256, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((51200, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 5, 32, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 16, 5), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 5, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 5, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1000, 7392), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2016), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 16384), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 128, 128, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 32, 128, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 64, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 64, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((160, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((640, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 640), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 640, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((160, 640), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 32, 32, 160), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 160, 32, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 32), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 32, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 256), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 96, 56, 56), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 56, 96, 56), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 7, 8, 7, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((288, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((64, 49, 3, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 64, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 64, 49, 3, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((49, 49, 3), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((64, 3, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 8, 8, 7, 7, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((384, 96), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((96, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 7, 4, 7, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((576, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((16, 49, 3, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 16, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 16, 49, 6, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((96, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((49, 49, 6), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((16, 6, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((96, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 4, 4, 7, 7, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((768, 192), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 7, 2, 7, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((1152, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((4, 49, 3, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 4, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 4, 49, 12, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((49, 49, 12), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((4, 12, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((48, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 2, 2, 7, 7, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((1536, 384), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 1536), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 7, 1, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((2304, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision", "pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose6,
        [((1, 49, 3, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-3"},
        },
    ),
    (
        Transpose5,
        [((3, 49, 1, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose0,
        [((3, 1, 49, 24, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((49, 49, 24), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((24, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 24, 49, 32), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((24, 32, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose5,
        [((1, 1, 1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-3"},
        },
    ),
    (
        Transpose1,
        [((3072, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 3072), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 7, 7, 768), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 7, 7), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 25088), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((4096, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_vgg13_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 12, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 12, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 49), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 50, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3072, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision", "pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose7,
        [((1, 50, 1, 3, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((50, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose3,
        [((1, 16, 50, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((50, 16, 1, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1024, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((50, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((4096, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 6400), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 1600), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 400), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 3600), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 900), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 85, 225), torch.float32)],
        {
            "model_names": ["pt_yolo_v5_yolov5n_img_cls_torchhub_480x480"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 16, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 85, 8400), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolox_yolox_l_obj_det_torchhub",
                "pt_yolox_yolox_m_obj_det_torchhub",
                "pt_yolox_yolox_darknet_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 16384), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose8,
        [((1, 1, 256, 64), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 16384), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 128, 128, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 512), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 4096), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 64, 64, 128), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1024), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 64, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1024), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 32, 32, 320), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 32, 32), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 64), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 16, 16, 512), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 16, 16), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 768), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 768), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 768), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 768), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 9, 12, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 9), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 27, 27, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 27, 16, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((197, 197, 16), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_large_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose3,
        [((1, 16, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-4", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((18, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet121_hf_xray_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2560), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_ghostnet_ghostnet_100_in1k_img_cls_timm",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv3_mobilenetv3_large_100_img_cls_timm",
                "pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm",
                "pt_mobilenetv2_basic_img_cls_torchhub",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((384, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((196, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 49), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 49, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 49), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((49, 256), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1001, 768), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1001, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2560, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 11, 32, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 11), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 32, 11, 80), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((10240, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2560, 10240), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 2560), torch.float32)],
        {
            "model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 6, 16, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 32, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2816, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1024, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((151936, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1536, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 12, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 35), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 35, 2, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 35, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8960, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1536, 8960), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((151936, 1536), torch.float32)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 29, 12, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 29, 2, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1000, 3024), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 3712), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2520), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1008), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 96, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 64, 96, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 8, 8, 8, 8, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((288, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose6,
        [((64, 64, 3, 3, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose5,
        [((3, 64, 64, 3, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 64, 64, 3, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((192, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((512, 2), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((64, 64, 3), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((64, 3, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 3, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((192, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((96, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((384, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((96, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((192, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 4, 8, 4, 8, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((576, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose6,
        [((16, 64, 3, 6, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose5,
        [((3, 64, 16, 6, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 16, 64, 6, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((96, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((6, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((64, 64, 6), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((6, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((16, 6, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((16, 6, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((96, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((192, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 4, 4, 8, 8, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((768, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((192, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((384, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 2, 8, 2, 8, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((1152, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose6,
        [((4, 64, 3, 12, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose5,
        [((3, 64, 4, 12, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 4, 64, 12, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((48, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((64, 64, 12), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((4, 12, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((4, 12, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((48, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((384, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 2, 2, 8, 8, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((1536, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((384, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((768, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 1, 8, 1, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose1,
        [((2304, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose6,
        [((1, 64, 3, 24, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-5", "dim1": "-3"}},
    ),
    (
        Transpose5,
        [((3, 64, 1, 24, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose0,
        [((3, 1, 64, 24, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((24, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((64, 64, 24), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((24, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((24, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose5,
        [((1, 1, 1, 8, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-3"}},
    ),
    (
        Transpose4,
        [((1, 8, 8, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 768, 8, 8), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 513, 12, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((513, 513, 12), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 12, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 61, 12, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((61, 61, 12), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((12, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32128, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 3, 85, 100), torch.float32)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 2048), torch.float32)],
        {
            "model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2048, 128), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 128, 16, 128), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((16, 128, 128), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 16, 128, 128), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose4,
        [((1, 27, 27, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 27, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 27, 12, 27), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((197, 197, 12), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_beit_microsoft_beit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 192, 196), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 3, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1000, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 196, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((9, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_basic_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((896, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 29, 14, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 32, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((128, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 29, 2, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((14, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 14, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((4864, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((896, 4864), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((151936, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 64, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 1, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 16384), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 128, 128, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 64, 128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4096, 2, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 2, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((2, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 2, 4096, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((128, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 64, 64, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 1024, 5, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 5, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((5, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 5, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((5, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 5, 1024, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1280, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1280, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((320, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 32, 32, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 320, 32, 32), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((512, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 8, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((8, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 2048, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose4,
        [((1, 16, 16, 512), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 512, 16, 16), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 512), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 256, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 320), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 1024, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 128), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 4096, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 64), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 16384, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 12, 64, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((12, 64, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 768, 128), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 768), torch.float32)],
        {
            "model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose7,
        [((1, 197, 1, 3, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 12, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 12, 1, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 1, 768), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_b_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose7,
        [((1, 197, 1, 3, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-5", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 16, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 16, 1, 64), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((197, 1, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 256, 16, 64), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((16, 256, 64), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((16, 64, 256), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 16, 256, 64), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((256008, 1024), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 4, 17, 4480), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 17, 1120), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 17, 280), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 4, 5880), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1, 80, 5880), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose8,
        [((7, 7, 3, 64), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 7, 3, 7), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 3, 7, 7), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 64, 256), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((256, 1, 64, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 64, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 64, 64), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 1, 64, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 64, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((3, 3, 64, 64), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 3, 64, 3), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 64, 3, 3), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((64, 1, 256, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((64, 256, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 256, 512), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 1, 256, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 256, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 256, 128), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((128, 1, 256, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 256, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((3, 3, 128, 128), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((128, 3, 128, 3), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 128, 3, 3), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 128, 512), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 1, 128, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 128, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 512, 128), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((128, 1, 512, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((128, 512, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 512, 1024), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1024, 1, 512, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1024, 512, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 512, 256), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((256, 1, 512, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 512, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((3, 3, 256, 256), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((256, 3, 256, 3), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 256, 3, 3), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 256, 1024), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1024, 1, 256, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1024, 256, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 1024, 256), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((256, 1, 1024, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((256, 1024, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 1024, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((2048, 1, 1024, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2048, 1024, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 1024, 512), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 1, 1024, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 1024, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((3, 3, 512, 512), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 3, 512, 3), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 512, 3, 3), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 512, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((2048, 1, 512, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((2048, 512, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose8,
        [((1, 1, 2048, 512), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-4", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((512, 1, 2048, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((512, 2048, 1, 1), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((21128, 768), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((28996, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1024, 576), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 512), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1000, 1280), torch.float32)],
        {
            "model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((30522, 768), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_mlm_padlenlp"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 192, 196), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 197, 3, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 3, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((3, 64, 197), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((3, 197, 64), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 192), torch.float32)],
        {
            "model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 4, 16, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((1, 288, 25), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 25, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 288), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((192, 48), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((25, 2, 1, 48), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((192, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((25, 1, 96), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((128, 784), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((64, 128), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 64), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((3, 12), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((12, 3), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((64, 12), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((128, 64), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((784, 128), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((9, 1024), torch.float32)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((51200, 1024), torch.float32)],
        {
            "model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1920), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((1000, 1408), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose1,
        [((768, 2304), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 256, 12, 64), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 256, 64), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-2", "dim1": "-1"},
        },
    ),
    (
        Transpose0,
        [((1, 12, 256, 64), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((50257, 768), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((50272, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 32, 12, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((12, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 12, 32, 64), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf", "pt_phi1_microsoft_phi_1_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim0": "-3", "dim1": "-2"},
        },
    ),
    (
        Transpose1,
        [((2, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 16, 12), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32, 12, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((512, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 513, 8, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((513, 513, 8), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 61, 8, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose4,
        [((61, 61, 8), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((1, 8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose0,
        [((1, 8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-3", "dim1": "-2"}},
    ),
    (
        Transpose1,
        [((8, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((2048, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
    (
        Transpose1,
        [((32128, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim0": "-2", "dim1": "-1"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Transpose")

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
