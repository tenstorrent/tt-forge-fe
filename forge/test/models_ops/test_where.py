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


class Where0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where0_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, where_input_0, where_input_2):
        where_output_1 = forge.op.Where("", where_input_0, self.get_constant("where0_const_1"), where_input_2)
        return where_output_1


class Where1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, where_input_0, where_input_1, where_input_2):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, where_input_2)
        return where_output_1


class Where2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where2_const_1", shape=(1,), dtype=torch.float32)
        self.add_constant("where2_const_2", shape=(1, 1, 256, 256), dtype=torch.float32)

    def forward(self, where_input_0):
        where_output_1 = forge.op.Where(
            "", where_input_0, self.get_constant("where2_const_1"), self.get_constant("where2_const_2")
        )
        return where_output_1


class Where3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where3_const_1", shape=(1,), dtype=torch.float32)
        self.add_constant("where3_const_2", shape=(1,), dtype=torch.float32)

    def forward(self, where_input_0):
        where_output_1 = forge.op.Where(
            "", where_input_0, self.get_constant("where3_const_1"), self.get_constant("where3_const_2")
        )
        return where_output_1


class Where4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where4_const_1", shape=(1,), dtype=torch.float32)
        self.add_constant("where4_const_2", shape=(1, 1, 7, 7), dtype=torch.float32)

    def forward(self, where_input_0):
        where_output_1 = forge.op.Where(
            "", where_input_0, self.get_constant("where4_const_1"), self.get_constant("where4_const_2")
        )
        return where_output_1


class Where5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where5_const_2", shape=(2441216,), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where5_const_2"))
        return where_output_1


class Where6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where6_const_2", shape=(1, 256, 6, 20), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where6_const_2"))
        return where_output_1


class Where7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where7_const_2", shape=(1, 256, 12, 40), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where7_const_2"))
        return where_output_1


class Where8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where8_const_2", shape=(1, 128, 12, 40), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where8_const_2"))
        return where_output_1


class Where9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where9_const_2", shape=(1, 128, 24, 80), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where9_const_2"))
        return where_output_1


class Where10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where10_const_2", shape=(1, 64, 24, 80), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where10_const_2"))
        return where_output_1


class Where11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where11_const_2", shape=(1, 64, 48, 160), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where11_const_2"))
        return where_output_1


class Where12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where12_const_2", shape=(1, 32, 48, 160), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where12_const_2"))
        return where_output_1


class Where13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where13_const_2", shape=(1, 32, 96, 320), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where13_const_2"))
        return where_output_1


class Where14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where14_const_2", shape=(1, 16, 96, 320), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where14_const_2"))
        return where_output_1


class Where15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where15_const_2", shape=(1, 16, 192, 640), dtype=torch.bfloat16)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where15_const_2"))
        return where_output_1


class Where16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("where16_const_2", shape=(1,), dtype=torch.float32)

    def forward(self, where_input_0, where_input_1):
        where_output_1 = forge.op.Where("", where_input_0, where_input_1, self.get_constant("where16_const_2"))
        return where_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Where0,
        [((1, 1, 128, 128), torch.bool), ((1, 1, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "onnx_albert_xlarge_v2_mlm_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_large_v2_mlm_hf",
                "onnx_albert_base_v2_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_albert_xxlarge_v2_mlm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Where1,
        [((1,), torch.bool), ((1,), torch.int64), ((1,), torch.int64)],
        {"model_names": ["onnx_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Where0,
        [((1, 1, 256, 256), torch.bool), ((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Where2,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Where3,
        [((1, 1, 5, 5), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_1_3b_seq_cls_hf",
                "pt_gptneo_gpt_neo_125m_seq_cls_hf",
                "pt_gptneo_gpt_neo_2_7b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Where3,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_names": [
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_gpt_neo_2_7b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Where0,
        [((1, 1, 7, 7), torch.bool), ((1, 1, 7, 7), torch.float32)],
        {"model_names": ["onnx_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Where4,
        [((1, 1, 7, 7), torch.bool)],
        {"model_names": ["onnx_nanogpt_financialsupport_nanogpt_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Where0,
        [((1, 1, 6, 6), torch.bool), ((1, 1, 6, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Where0,
        [((2, 1, 7, 7), torch.bool), ((2, 1, 7, 7), torch.float32)],
        {"model_names": ["onnx_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Where5,
        [((2441216,), torch.bool), ((2441216,), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Where1,
        [((2441216,), torch.bool), ((2441216,), torch.float32), ((2441216,), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Where6,
        [((1, 256, 6, 20), torch.bool), ((1, 256, 6, 20), torch.bfloat16)],
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
        Where1,
        [((1, 256, 6, 20), torch.bool), ((1, 256, 6, 20), torch.bfloat16), ((1, 256, 6, 20), torch.bfloat16)],
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
        Where7,
        [((1, 256, 12, 40), torch.bool), ((1, 256, 12, 40), torch.bfloat16)],
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
        Where1,
        [((1, 256, 12, 40), torch.bool), ((1, 256, 12, 40), torch.bfloat16), ((1, 256, 12, 40), torch.bfloat16)],
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
        Where8,
        [((1, 128, 12, 40), torch.bool), ((1, 128, 12, 40), torch.bfloat16)],
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
        Where1,
        [((1, 128, 12, 40), torch.bool), ((1, 128, 12, 40), torch.bfloat16), ((1, 128, 12, 40), torch.bfloat16)],
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
        Where9,
        [((1, 128, 24, 80), torch.bool), ((1, 128, 24, 80), torch.bfloat16)],
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
        Where1,
        [((1, 128, 24, 80), torch.bool), ((1, 128, 24, 80), torch.bfloat16), ((1, 128, 24, 80), torch.bfloat16)],
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
        Where10,
        [((1, 64, 24, 80), torch.bool), ((1, 64, 24, 80), torch.bfloat16)],
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
        Where1,
        [((1, 64, 24, 80), torch.bool), ((1, 64, 24, 80), torch.bfloat16), ((1, 64, 24, 80), torch.bfloat16)],
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
        Where11,
        [((1, 64, 48, 160), torch.bool), ((1, 64, 48, 160), torch.bfloat16)],
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
        Where1,
        [((1, 64, 48, 160), torch.bool), ((1, 64, 48, 160), torch.bfloat16), ((1, 64, 48, 160), torch.bfloat16)],
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
        Where12,
        [((1, 32, 48, 160), torch.bool), ((1, 32, 48, 160), torch.bfloat16)],
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
        Where1,
        [((1, 32, 48, 160), torch.bool), ((1, 32, 48, 160), torch.bfloat16), ((1, 32, 48, 160), torch.bfloat16)],
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
        Where13,
        [((1, 32, 96, 320), torch.bool), ((1, 32, 96, 320), torch.bfloat16)],
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
        Where1,
        [((1, 32, 96, 320), torch.bool), ((1, 32, 96, 320), torch.bfloat16), ((1, 32, 96, 320), torch.bfloat16)],
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
        Where14,
        [((1, 16, 96, 320), torch.bool), ((1, 16, 96, 320), torch.bfloat16)],
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
        Where1,
        [((1, 16, 96, 320), torch.bool), ((1, 16, 96, 320), torch.bfloat16), ((1, 16, 96, 320), torch.bfloat16)],
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
        Where15,
        [((1, 16, 192, 640), torch.bool), ((1, 16, 192, 640), torch.bfloat16)],
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
        Where1,
        [((1, 16, 192, 640), torch.bool), ((1, 16, 192, 640), torch.bfloat16), ((1, 16, 192, 640), torch.bfloat16)],
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
        Where16,
        [((1, 1, 256), torch.bool), ((1, 1, 256), torch.float32)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Where")

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
