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


class Subtract0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract0_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract0_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, subtract_input_0, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, subtract_input_1)
        return subtract_output_1


class Subtract2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract2_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract2_const_1"))
        return subtract_output_1


class Subtract3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract3_const_0", shape=(1, 2, 8400), dtype=torch.bfloat16)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract3_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract4_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract4_const_1"))
        return subtract_output_1


class Subtract5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract5_const_0", shape=(1, 2, 8400), dtype=torch.float32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract5_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract6_const_0", shape=(5880, 2), dtype=torch.bfloat16)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract6_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract7_const_0", shape=(1,), dtype=torch.int32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract7_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract8_const_0", shape=(1,), dtype=torch.bfloat16)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract8_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract9_const_1", shape=(1,), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract9_const_1"))
        return subtract_output_1


class Subtract10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract10_const_1", shape=(1, 256, 6, 20), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract10_const_1"))
        return subtract_output_1


class Subtract11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract11_const_1", shape=(1, 256, 12, 40), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract11_const_1"))
        return subtract_output_1


class Subtract12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract12_const_1", shape=(1, 128, 12, 40), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract12_const_1"))
        return subtract_output_1


class Subtract13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract13_const_1", shape=(1, 128, 24, 80), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract13_const_1"))
        return subtract_output_1


class Subtract14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract14_const_1", shape=(1, 64, 24, 80), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract14_const_1"))
        return subtract_output_1


class Subtract15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract15_const_1", shape=(1, 64, 48, 160), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract15_const_1"))
        return subtract_output_1


class Subtract16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract16_const_1", shape=(1, 32, 48, 160), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract16_const_1"))
        return subtract_output_1


class Subtract17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract17_const_1", shape=(1, 32, 96, 320), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract17_const_1"))
        return subtract_output_1


class Subtract18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract18_const_1", shape=(1, 16, 96, 320), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract18_const_1"))
        return subtract_output_1


class Subtract19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract19_const_1", shape=(1, 16, 192, 640), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract19_const_1"))
        return subtract_output_1


class Subtract20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract20_const_1", shape=(3, 1, 1), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract20_const_1"))
        return subtract_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Subtract0,
        [((1, 1, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "onnx_albert_xlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_large_v1_mlm_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "onnx_albert_large_v2_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_qwen_v3_0_6b_clm_hf",
                "onnx_albert_base_v1_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_qwen_v3_1_7b_clm_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "onnx_albert_xlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_qwen_v3_4b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
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
        Subtract1,
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
        Subtract1,
        [((1, 12, 6625), torch.float32), ((1, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (Subtract0, [((1, 1, 14, 14), torch.float32)], {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99}),
    (
        Subtract0,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_xglm_564m_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract2,
        [((1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_bloom_default_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_bloom_default_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 1, 2048), torch.float32)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "onnx_perceiverio_deepmind_language_perceiver_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 1, 1, 25), torch.float32)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract3,
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
        Subtract1,
        [((1, 2, 8400), torch.bfloat16), ((1, 2, 8400), torch.bfloat16)],
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
        Subtract4,
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
        Subtract0,
        [((1, 256, 6, 20), torch.float32)],
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
        Subtract0,
        [((1, 256, 12, 40), torch.float32)],
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
        Subtract0,
        [((1, 128, 12, 40), torch.float32)],
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
        Subtract0,
        [((1, 128, 24, 80), torch.float32)],
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
        Subtract0,
        [((1, 64, 24, 80), torch.float32)],
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
        Subtract0,
        [((1, 64, 48, 160), torch.float32)],
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
        Subtract0,
        [((1, 32, 48, 160), torch.float32)],
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
        Subtract0,
        [((1, 32, 96, 320), torch.float32)],
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
        Subtract0,
        [((1, 16, 96, 320), torch.float32)],
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
        Subtract0,
        [((1, 16, 192, 640), torch.float32)],
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
        Subtract5,
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
        Subtract1,
        [((1, 2, 8400), torch.float32), ((1, 2, 8400), torch.float32)],
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
        Subtract1,
        [((1, 14, 768), torch.float32), ((1, 14, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 12, 14, 14), torch.float32), ((1, 12, 14, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_qa_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
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
        Subtract1,
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
        Subtract1,
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
        Subtract1,
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
        Subtract1,
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
        Subtract1,
        [((1, 25, 97), torch.float32), ((1, 25, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 1, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
                "onnx_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
        [((1024, 72), torch.float32), ((1024, 72), torch.float32)],
        {
            "model_names": [
                "pt_nbeats_generic_basis_time_series_forecasting_github",
                "pt_nbeats_trend_basis_time_series_forecasting_github",
                "pt_nbeats_seasonality_basis_time_series_forecasting_github",
                "onnx_nbeats_seasionality_basis_time_series_forecasting_github",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
        [((1, 12, 97), torch.float32), ((1, 12, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 1, 5, 5), torch.float32)],
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
        Subtract0,
        [((1, 1, 384, 384), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract2,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Subtract6,
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
    (
        Subtract1,
        [((1, 5880, 2), torch.bfloat16), ((1, 5880, 2), torch.bfloat16)],
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
    (
        Subtract0,
        [((1, 1, 1, 201), torch.float32)],
        {"model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 11, 128), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
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
        Subtract1,
        [((1, 11, 312), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 15, 768), torch.float32), ((1, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 12, 15, 15), torch.float32), ((1, 12, 15, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
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
        Subtract1,
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
        Subtract0,
        [((1, 1, 25, 25), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract4,
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
        Subtract0,
        [((1, 256, 10, 32), torch.float32)],
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
        Subtract0,
        [((1, 256, 20, 64), torch.float32)],
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
        Subtract0,
        [((1, 128, 20, 64), torch.float32)],
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
        Subtract0,
        [((1, 128, 40, 128), torch.float32)],
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
        Subtract0,
        [((1, 64, 40, 128), torch.float32)],
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
        Subtract0,
        [((1, 64, 80, 256), torch.float32)],
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
        Subtract0,
        [((1, 32, 80, 256), torch.float32)],
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
        Subtract0,
        [((1, 32, 160, 512), torch.float32)],
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
        Subtract0,
        [((1, 16, 160, 512), torch.float32)],
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
        Subtract0,
        [((1, 16, 320, 1024), torch.float32)],
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
        Subtract1,
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
        Subtract1,
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
        Subtract1,
        [((4, 5, 768), torch.float32), ((4, 5, 1), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
        [((4, 12, 5, 5), torch.float32), ((4, 12, 5, 1), torch.float32)],
        {
            "model_names": ["pd_chineseclip_ofa_sys_chinese_clip_vit_base_patch16_img_text_pairing_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
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
    (Subtract7, [((1, 11), torch.int32)], {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99}),
    (
        Subtract0,
        [((1, 1, 6, 6), torch.float32)],
        {
            "model_names": [
                "pt_qwen1_5_0_5b_clm_hf",
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 16, 16), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (Subtract7, [((1, 1, 32, 32), torch.int32)], {"model_names": ["pt_bloom_default_clm_hf"], "pcc": 0.99}),
    (
        Subtract8,
        [((1, 1, 1, 201), torch.bfloat16)],
        {"model_names": ["pt_vilt_vqa_qa_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract1,
        [((1, 112, 112, 64), torch.float32), ((1, 1, 1, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 56, 55, 64), torch.float32), ((1, 1, 1, 64), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 56, 55, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 56, 55, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 28, 28, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 28, 28, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 28, 28, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 14, 14, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 14, 14, 1024), torch.float32), ((1, 1, 1, 1024), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 14, 14, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 7, 7, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 7, 7, 2048), torch.float32), ((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["jax_resnet_50_img_cls_hf"], "pcc": 0.99},
    ),
    (Subtract0, [((1, 1, 10, 10), torch.float32)], {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99}),
    (
        Subtract1,
        [((1, 25, 6625), torch.float32), ((1, 25, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((2, 1, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
                "onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract8,
        [((1, 1, 1, 204), torch.bfloat16)],
        {"model_names": ["pt_vilt_mlm_mlm_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract1,
        [((1, 10, 768), torch.float32), ((1, 10, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 12, 10, 10), torch.float32), ((1, 12, 10, 1), torch.float32)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Subtract0, [((1, 1, 9, 9), torch.float32)], {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99}),
    (
        Subtract1,
        [((2, 4, 768), torch.float32), ((2, 4, 1), torch.float32)],
        {"model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((2, 12, 4, 4), torch.float32), ((2, 12, 4, 1), torch.float32)],
        {"model_names": ["pd_blip_salesforce_blip_image_captioning_base_img_captioning_padlenlp"], "pcc": 0.99},
    ),
    (Subtract0, [((1, 1, 39, 39), torch.float32)], {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99}),
    (
        Subtract1,
        [((1, 1, 1, 64), torch.float32), ((1, 1, 1, 64), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 1, 1, 256), torch.float32), ((1, 1, 1, 256), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 1, 1, 512), torch.float32), ((1, 1, 1, 512), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 1, 1, 128), torch.float32), ((1, 1, 1, 128), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 1, 1, 1024), torch.float32), ((1, 1, 1, 1024), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 1, 1, 2048), torch.float32), ((1, 1, 1, 2048), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99},
    ),
    (Subtract4, [((2441216,), torch.float32)], {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99}),
    (Subtract7, [((1, 9), torch.int32)], {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99}),
    (
        Subtract1,
        [((2, 32, 10, 16384), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 10, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 20, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 20, 1024), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 40, 1024), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 80, 1024), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 60, 1024), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 60, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 40, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 30, 4096), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 30, 16384), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((2, 32, 20, 16384), torch.bfloat16), ((2, 32, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract9,
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
        Subtract10,
        [((1, 256, 6, 20), torch.bfloat16)],
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
        Subtract11,
        [((1, 256, 12, 40), torch.bfloat16)],
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
        Subtract12,
        [((1, 128, 12, 40), torch.bfloat16)],
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
        Subtract13,
        [((1, 128, 24, 80), torch.bfloat16)],
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
        Subtract14,
        [((1, 64, 24, 80), torch.bfloat16)],
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
        Subtract15,
        [((1, 64, 48, 160), torch.bfloat16)],
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
        Subtract16,
        [((1, 32, 48, 160), torch.bfloat16)],
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
        Subtract17,
        [((1, 32, 96, 320), torch.bfloat16)],
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
        Subtract18,
        [((1, 16, 96, 320), torch.bfloat16)],
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
        Subtract19,
        [((1, 16, 192, 640), torch.bfloat16)],
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
        Subtract1,
        [((100, 8, 9240), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((100, 8, 4480), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((100, 8, 8640), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((100, 8, 17280), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((100, 8, 34240), torch.float32), ((100, 8, 1), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 64, 1, 1), torch.bfloat16), ((1, 64, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((1, 256, 1, 1), torch.bfloat16), ((1, 256, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((1, 128, 1, 1), torch.bfloat16), ((1, 128, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((1, 512, 1, 1), torch.bfloat16), ((1, 512, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((1, 1024, 1, 1), torch.bfloat16), ((1, 1024, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract1,
        [((1, 2048, 1, 1), torch.bfloat16), ((1, 2048, 1, 1), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract8,
        [((1, 1, 850, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract8,
        [((1, 1, 100, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract8,
        [((1, 1, 1, 25, 34), torch.bfloat16)],
        {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract1,
        [((100, 8, 33, 850), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract1,
        [((100, 8, 16, 850), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract1,
        [((100, 8, 8, 3350), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract1,
        [((100, 8, 4, 13400), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract1,
        [((100, 8, 2, 53400), torch.bfloat16), ((100, 8, 1, 1), torch.bfloat16)],
        {"model_names": ["pt_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract20,
        [((3, 480, 640), torch.bfloat16)],
        {
            "model_names": ["pt_ssd300_vgg16_ssd300_vgg16_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract0,
        [((1, 1, 356, 356), torch.float32)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Subtract0,
        [((1, 1, 512, 512), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Subtract")

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
