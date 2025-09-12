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


class Repeatinterleave0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=0)
        return repeatinterleave_output_1


class Repeatinterleave1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=1)
        return repeatinterleave_output_1


class Repeatinterleave2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=14, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=32, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=1, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=256, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=7, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=35, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=25, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=80, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=40, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=20, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=10, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=60, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=30, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=15, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=160, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=80, dim=3)
        return repeatinterleave_output_1


class Repeatinterleave18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=40, dim=3)
        return repeatinterleave_output_1


class Repeatinterleave19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=20, dim=3)
        return repeatinterleave_output_1


class Repeatinterleave20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=128, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=10, dim=3)
        return repeatinterleave_output_1


class Repeatinterleave22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=5, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=384, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=6, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=29, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=60, dim=3)
        return repeatinterleave_output_1


class Repeatinterleave27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=30, dim=3)
        return repeatinterleave_output_1


class Repeatinterleave28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=15, dim=3)
        return repeatinterleave_output_1


class Repeatinterleave29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=2, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=3, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=4, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=16, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=160, dim=3)
        return repeatinterleave_output_1


class Repeatinterleave34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=9, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=39, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=2, dim=0)
        return repeatinterleave_output_1


class Repeatinterleave37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=850, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=100, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=8, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=356, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=512, dim=2)
        return repeatinterleave_output_1


class Repeatinterleave42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, repeatinterleave_input_0):
        repeatinterleave_output_1 = forge.op.RepeatInterleave("", repeatinterleave_input_0, repeats=24, dim=2)
        return repeatinterleave_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Repeatinterleave0,
        [((1, 14), torch.int64)],
        {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 14), torch.int64)],
        {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 14), torch.int64)],
        {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave2,
        [((1, 1, 1, 14), torch.int64)],
        {"model_names": ["pt_albert_squad2_qa_hf"], "pcc": 0.99, "args": {"repeats": "14", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave3,
        [((1, 1, 1, 32), torch.int64)],
        {
            "model_names": [
                "pt_opt_facebook_opt_125m_seq_cls_hf",
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "32", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2048, 768), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 256, 1280), torch.float32)],
        {
            "model_names": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 16, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_phi_1_5_microsoft_phi_1_5_token_cls_hf",
                "pt_phi1_microsoft_phi_1_token_cls_hf",
                "pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_phi1_microsoft_phi_1_seq_cls_hf",
                "pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf",
                "pt_fuyu_adept_fuyu_8b_qa_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_token_cls_hf",
                "pt_phi2_microsoft_phi_2_seq_cls_hf",
                "pt_phi2_microsoft_phi_2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
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
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
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
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave5,
        [((1, 1, 1, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi_1_5_microsoft_phi_1_5_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
                "pt_bart_large_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_xglm_1_7b_clm_hf",
                "pt_gptneo_gpt_neo_125m_clm_hf",
                "pt_gptneo_gpt_neo_1_3b_clm_hf",
                "pt_phi1_microsoft_phi_1_clm_hf",
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
            "args": {"repeats": "256", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_0_5b_clm_hf",
                "pt_qwen1_5_0_5b_clm_hf",
                "pt_llama3_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen_v2_0_5b_instruct_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
                "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_qwen_coder_0_5b_clm_hf",
                "pt_qwen1_5_0_5b_clm_hf",
                "pt_llama3_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf",
                "pt_qwen1_5_0_5b_chat_clm_hf",
                "pt_qwen_v2_0_5b_instruct_clm_hf",
                "pt_llama3_llama_3_2_1b_clm_hf",
                "pt_llama3_llama_3_2_1b_instruct_clm_hf",
                "pt_falcon_tiiuae_falcon_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave6,
        [((1, 2, 1, 35, 64), torch.float32)],
        {"model_names": ["pt_qwen_coder_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "7", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 35), torch.int64)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 35), torch.int64)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave7,
        [((1, 1, 1, 35), torch.int64)],
        {
            "model_names": ["pt_qwen_coder_0_5b_clm_hf", "pt_qwen_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "35", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 25), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 25), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 1, 1, 25), torch.int64)],
        {
            "model_names": [
                "pt_stereo_medium_music_generation_hf",
                "pt_stereo_small_music_generation_hf",
                "pt_stereo_large_music_generation_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave8,
        [((1, 1, 1, 25), torch.int64)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"repeats": "25", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave9,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "80", "dim": "2"},
        },
    ),
    (
        Repeatinterleave10,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "40", "dim": "2"},
        },
    ),
    (
        Repeatinterleave11,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "20", "dim": "2"},
        },
    ),
    (
        Repeatinterleave12,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "10", "dim": "2"},
        },
    ),
    (
        Repeatinterleave13,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "60", "dim": "2"},
        },
    ),
    (
        Repeatinterleave14,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "30", "dim": "2"},
        },
    ),
    (
        Repeatinterleave15,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "15", "dim": "2"},
        },
    ),
    (
        Repeatinterleave16,
        [((1, 3, 1, 1, 2), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "160", "dim": "2"},
        },
    ),
    (
        Repeatinterleave17,
        [((1, 3, 80, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "80", "dim": "3"},
        },
    ),
    (
        Repeatinterleave18,
        [((1, 3, 40, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "40", "dim": "3"},
        },
    ),
    (
        Repeatinterleave19,
        [((1, 3, 20, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_640x640",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_640x640",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "20", "dim": "3"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_qwen_v3_1_7b_clm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_qwen_v3_4b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_qwen_v3_1_7b_clm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_qwen_v3_4b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave20,
        [((1, 1, 1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_qwen_v3_1_7b_clm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_qwen_v3_4b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "128", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_large_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_vit_vit_l_32_img_cls_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_vit_large_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_deepseek_1_3b_instruct_qa_hf",
                "pt_qwen_v2_1_5b_clm_hf",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_qwen_v3_1_7b_clm_hf",
                "pt_qwen_coder_1_5b_clm_hf",
                "pt_llava_1_5_7b_cond_gen_hf",
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
                "pt_deepseek_7b_instruct_qa_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_token_cls_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
                "pt_qwen_v3_4b_clm_hf",
                "pt_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 64, 1), torch.float32)],
        {
            "model_names": [
                "pt_deepseek_1_3b_instruct_qa_hf",
                "pt_qwen_v2_1_5b_clm_hf",
                "pt_qwen_v3_0_6b_clm_hf",
                "pt_qwen_v3_1_7b_clm_hf",
                "pt_qwen_coder_1_5b_clm_hf",
                "pt_llava_1_5_7b_cond_gen_hf",
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
                "pt_deepseek_7b_instruct_qa_hf",
                "pt_llama3_huggyllama_7b_clm_hf",
                "pt_llama3_huggyllama_7b_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_seq_cls_hf",
                "pt_phi4_microsoft_phi_4_token_cls_hf",
                "pt_qwen_v2_qwen_qwen2_7b_token_cls_hf",
                "pt_qwen_v3_4b_clm_hf",
                "pt_cogito_v1_preview_llama_3b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave6,
        [((1, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_nanogpt_financialsupport_nanogpt_text_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave21,
        [((1, 3, 10, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5s_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5m_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_320x320",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_320x320",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "10", "dim": "3"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 5), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 5), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave22,
        [((1, 1, 1, 5), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "5", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 384), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 384), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave23,
        [((1, 1, 1, 384), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "384", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 512, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave24,
        [((1, 2, 1, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_1_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "6", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 29), torch.int64)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 29), torch.int64)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave25,
        [((1, 1, 1, 29), torch.int64)],
        {
            "model_names": ["pt_qwen_v2_1_5b_clm_hf", "pt_qwen1_5_0_5b_chat_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "29", "dim": "2"},
        },
    ),
    (
        Repeatinterleave26,
        [((1, 3, 60, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "60", "dim": "3"},
        },
    ),
    (
        Repeatinterleave27,
        [((1, 3, 30, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "30", "dim": "3"},
        },
    ),
    (
        Repeatinterleave28,
        [((1, 3, 15, 1, 2), torch.bfloat16)],
        {
            "model_names": [
                "pt_yolo_v5_yolov5m_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5n_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5s_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5x_img_cls_torchhub_480x480",
                "pt_yolo_v5_yolov5l_img_cls_torchhub_480x480",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "15", "dim": "3"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 25), torch.int64)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 128, 1), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_gemma_google_gemma_1_1_2b_it_qa_hf",
                "pt_gemma_google_gemma_1_1_7b_it_qa_hf",
                "pt_gemma_google_gemma_2b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 128, 1), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
                "pt_gemma_google_gemma_1_1_2b_it_qa_hf",
                "pt_gemma_google_gemma_1_1_7b_it_qa_hf",
                "pt_gemma_google_gemma_2b_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 4, 1, 522, 256), torch.float32)],
        {
            "model_names": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave29,
        [((1, 4, 1, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "2", "dim": "2"},
        },
    ),
    (
        Repeatinterleave30,
        [((1, 4, 1, 522, 256), torch.float32)],
        {
            "model_names": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_tiny_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 6), torch.int64)],
        {"model_names": ["pt_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 6), torch.int64)],
        {"model_names": ["pt_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave24,
        [((1, 1, 1, 6), torch.int64)],
        {"model_names": ["pt_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "6", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 64, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_qwen_v3_embedding_4b_sentence_embed_gen_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 64, 1), torch.bfloat16)],
        {
            "model_names": [
                "pt_qwen_v3_embedding_4b_sentence_embed_gen_hf",
                "pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave31,
        [((4, 8, 1, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_4b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave29,
        [((4, 8, 1, 31, 128), torch.bfloat16)],
        {
            "model_names": ["pt_qwen_v3_embedding_0_6b_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "2", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_base_img_cls_hf",
                "pt_deit_base_distilled_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
                "pt_deit_base_img_cls_hf",
                "pt_vit_vit_b_32_img_cls_torchvision",
                "pt_beit_base_img_cls_hf",
                "pt_mgp_default_scene_text_recognition_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave32,
        [((1, 1, 1, 16), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "16", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1280), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_h_14_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave31,
        [((1, 8, 1, 4, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_seq_cls_hf", "pt_llama3_llama_3_2_1b_instruct_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 128, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf", "pt_qwen_v3_4b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave29,
        [((1, 8, 1, 128, 128), torch.float32)],
        {
            "model_names": ["pt_qwen_v3_0_6b_clm_hf", "pt_qwen_v3_1_7b_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "2", "dim": "2"},
        },
    ),
    (
        Repeatinterleave31,
        [((1, 8, 1, 128, 128), torch.float32)],
        {"model_names": ["pt_qwen_v3_4b_clm_hf"], "pcc": 0.99, "args": {"repeats": "4", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 10), torch.int64)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 10), torch.int64)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 10), torch.int64)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave12,
        [((1, 1, 1, 10), torch.int64)],
        {"model_names": ["pt_roberta_xlm_base_mlm_hf"], "pcc": 0.99, "args": {"repeats": "10", "dim": "2"}},
    ),
    (
        Repeatinterleave1,
        [((2, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave6,
        [((2, 1, 1, 7), torch.int64)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"repeats": "7", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 50176, 256), torch.bfloat16)],
        {
            "model_names": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave33,
        [((1, 3, 160, 1, 2), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "160", "dim": "3"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 9), torch.int64)],
        {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 9), torch.int64)],
        {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 9), torch.int64)],
        {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave34,
        [((1, 1, 1, 9), torch.int64)],
        {"model_names": ["pt_albert_imdb_seq_cls_hf"], "pcc": 0.99, "args": {"repeats": "9", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave24,
        [((1, 2, 1, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"repeats": "6", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 2, 1, 39, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave6,
        [((1, 2, 1, 39, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"repeats": "7", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave35,
        [((1, 1, 1, 39), torch.int64)],
        {"model_names": ["pt_qwen_v2_0_5b_instruct_clm_hf"], "pcc": 0.99, "args": {"repeats": "39", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1024), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1024), torch.float32)],
        {"model_names": ["pt_llava_1_5_7b_cond_gen_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave31,
        [((1, 8, 1, 12, 128), torch.float32)],
        {
            "model_names": [
                "pt_ministral_ministral_3b_instruct_clm_hf",
                "pt_ministral_ministral_8b_instruct_clm_hf",
                "pt_mistral_7b_clm_hf",
                "pt_mistral_7b_instruct_v03_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 48, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave4,
        [((1, 48, 1), torch.float32)],
        {
            "model_names": [
                "pt_phi3_5_mini_instruct_clm_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_128k_instruct_token_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf",
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 384), torch.bfloat16)],
        {
            "model_names": ["pt_deit_small_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    pytest.param(
        (
            Repeatinterleave36,
            [((1,), torch.bfloat16)],
            {
                "model_names": ["pt_stable_diffusion_stable_diffusion_xl_base_1_0_cond_gen_hf"],
                "pcc": 0.99,
                "default_df_override": "Float16_b",
                "args": {"repeats": "2", "dim": "0"},
            },
        ),
        marks=[
            pytest.mark.xfail(
                reason="RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/common/shape.cpp:55: normalized_index >= 0 and normalized_index < rank"
            )
        ],
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave37,
        [((1, 1, 1, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "850", "dim": "2"},
        },
    ),
    (
        Repeatinterleave38,
        [((1, 1, 1, 850), torch.bfloat16)],
        {
            "model_names": ["pt_detr_resnet_50_obj_det_hf", "pt_detr_resnet_50_panoptic_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"repeats": "100", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_clm_hf", "pt_llama3_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave31,
        [((1, 8, 1, 256, 64), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_1b_clm_hf", "pt_llama3_llama_3_2_1b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 356, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 356, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave39,
        [((1, 1, 1, 356, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99, "args": {"repeats": "8", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 356), torch.int64)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 356), torch.int64)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "1"},
        },
    ),
    (
        Repeatinterleave40,
        [((1, 1, 1, 356), torch.int64)],
        {
            "model_names": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"],
            "pcc": 0.99,
            "args": {"repeats": "356", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 512, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 512, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave39,
        [((1, 1, 1, 512, 256), torch.float32)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "8", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 512), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 512), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave41,
        [((1, 1, 1, 512), torch.int64)],
        {"model_names": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99, "args": {"repeats": "512", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_2_3b_clm_hf",
                "pt_llama3_llama_3_2_3b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave31,
        [((1, 8, 1, 256, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_clm_hf",
                "pt_llama3_llama_3_1_8b_instruct_clm_hf",
                "pt_llama3_llama_3_8b_clm_hf",
                "pt_llama3_llama_3_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave30,
        [((1, 8, 1, 256, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_clm_hf", "pt_llama3_llama_3_2_3b_instruct_clm_hf"],
            "pcc": 0.99,
            "args": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave31,
        [((1, 8, 1, 4, 128), torch.float32)],
        {
            "model_names": [
                "pt_llama3_llama_3_1_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_llama_3_8b_seq_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"repeats": "4", "dim": "2"},
        },
    ),
    (
        Repeatinterleave30,
        [((1, 8, 1, 4, 128), torch.float32)],
        {
            "model_names": ["pt_llama3_llama_3_2_3b_instruct_seq_cls_hf", "pt_llama3_llama_3_2_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 10, 1, 5, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave31,
        [((1, 10, 1, 5, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_seq_cls_hf"], "pcc": 0.99, "args": {"repeats": "4", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 10, 1, 12, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave31,
        [((1, 10, 1, 12, 128), torch.float32)],
        {"model_names": ["pt_phi4_microsoft_phi_4_token_cls_hf"], "pcc": 0.99, "args": {"repeats": "4", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 4, 1, 13, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave6,
        [((1, 4, 1, 13, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99, "args": {"repeats": "7", "dim": "2"}},
    ),
    (
        Repeatinterleave0,
        [((1, 8, 1, 44, 128), torch.float32)],
        {
            "model_names": ["pt_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "1", "dim": "0"},
        },
    ),
    (
        Repeatinterleave30,
        [((1, 8, 1, 44, 128), torch.float32)],
        {
            "model_names": ["pt_cogito_v1_preview_llama_3b_text_gen_hf"],
            "pcc": 0.99,
            "args": {"repeats": "3", "dim": "2"},
        },
    ),
    (
        Repeatinterleave0,
        [((1, 1, 1, 24), torch.int64)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "0"}},
    ),
    (
        Repeatinterleave1,
        [((1, 1, 1, 24), torch.int64)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "1"}},
    ),
    (
        Repeatinterleave42,
        [((1, 1, 1, 24), torch.int64)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"repeats": "24", "dim": "2"}},
    ),
    (
        Repeatinterleave4,
        [((1, 1, 1, 24), torch.int64)],
        {"model_names": ["pt_speecht5_tts_tts_text_to_speech_hf"], "pcc": 0.99, "args": {"repeats": "1", "dim": "2"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("RepeatInterleave")

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
