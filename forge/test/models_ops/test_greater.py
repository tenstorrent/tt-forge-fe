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


class Greater0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater0_const_1", shape=(1, 256, 10, 32), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater0_const_1"))
        return greater_output_1


class Greater1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater1_const_1", shape=(1, 256, 20, 64), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater1_const_1"))
        return greater_output_1


class Greater2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater2_const_1", shape=(1, 128, 20, 64), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater2_const_1"))
        return greater_output_1


class Greater3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater3_const_1", shape=(1, 128, 40, 128), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater3_const_1"))
        return greater_output_1


class Greater4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater4_const_1", shape=(1, 64, 40, 128), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater4_const_1"))
        return greater_output_1


class Greater5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater5_const_1", shape=(1, 64, 80, 256), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater5_const_1"))
        return greater_output_1


class Greater6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater6_const_1", shape=(1, 32, 80, 256), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater6_const_1"))
        return greater_output_1


class Greater7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater7_const_1", shape=(1, 32, 160, 512), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater7_const_1"))
        return greater_output_1


class Greater8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater8_const_1", shape=(1, 16, 160, 512), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater8_const_1"))
        return greater_output_1


class Greater9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater9_const_1", shape=(1, 16, 320, 1024), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater9_const_1"))
        return greater_output_1


class Greater10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater10_const_1", shape=(1,), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater10_const_1"))
        return greater_output_1


class Greater11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater11_const_1", shape=(1, 256, 6, 20), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater11_const_1"))
        return greater_output_1


class Greater12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater12_const_1", shape=(1, 256, 12, 40), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater12_const_1"))
        return greater_output_1


class Greater13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater13_const_1", shape=(1, 128, 12, 40), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater13_const_1"))
        return greater_output_1


class Greater14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater14_const_1", shape=(1, 128, 24, 80), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater14_const_1"))
        return greater_output_1


class Greater15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater15_const_1", shape=(1, 64, 24, 80), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater15_const_1"))
        return greater_output_1


class Greater16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater16_const_1", shape=(1, 64, 48, 160), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater16_const_1"))
        return greater_output_1


class Greater17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater17_const_1", shape=(1, 32, 48, 160), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater17_const_1"))
        return greater_output_1


class Greater18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater18_const_1", shape=(1, 32, 96, 320), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater18_const_1"))
        return greater_output_1


class Greater19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater19_const_1", shape=(1, 16, 96, 320), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater19_const_1"))
        return greater_output_1


class Greater20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater20_const_1", shape=(1, 16, 192, 640), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater20_const_1"))
        return greater_output_1


class Greater21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater21_const_1", shape=(1, 6, 2048), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater21_const_1"))
        return greater_output_1


class Greater22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater22_const_1", shape=(1, 32, 480, 640), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater22_const_1"))
        return greater_output_1


class Greater23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater23_const_1", shape=(1, 64, 240, 320), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater23_const_1"))
        return greater_output_1


class Greater24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater24_const_1", shape=(1, 32, 240, 320), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater24_const_1"))
        return greater_output_1


class Greater25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater25_const_1", shape=(1, 128, 120, 160), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater25_const_1"))
        return greater_output_1


class Greater26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater26_const_1", shape=(1, 64, 120, 160), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater26_const_1"))
        return greater_output_1


class Greater27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater27_const_1", shape=(1, 256, 60, 80), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater27_const_1"))
        return greater_output_1


class Greater28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater28_const_1", shape=(1, 128, 60, 80), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater28_const_1"))
        return greater_output_1


class Greater29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater29_const_1", shape=(1, 512, 30, 40), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater29_const_1"))
        return greater_output_1


class Greater30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater30_const_1", shape=(1, 256, 30, 40), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater30_const_1"))
        return greater_output_1


class Greater31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater31_const_1", shape=(1, 1024, 15, 20), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater31_const_1"))
        return greater_output_1


class Greater32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater32_const_1", shape=(1, 512, 15, 20), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater32_const_1"))
        return greater_output_1


class Greater33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("greater33_const_1", shape=(1, 6, 3072), dtype=torch.float32)

    def forward(self, greater_input_0):
        greater_output_1 = forge.op.Greater("", greater_input_0, self.get_constant("greater33_const_1"))
        return greater_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Greater0,
        [((1, 256, 10, 32), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater1,
        [((1, 256, 20, 64), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater2,
        [((1, 128, 20, 64), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater3,
        [((1, 128, 40, 128), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater4,
        [((1, 64, 40, 128), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater5,
        [((1, 64, 80, 256), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater6,
        [((1, 32, 80, 256), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater7,
        [((1, 32, 160, 512), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater8,
        [((1, 16, 160, 512), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater9,
        [((1, 16, 320, 1024), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision",
                "pt_monodepth2_mono_1024x320_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater10,
        [((1, 1, 32, 32), torch.float32)],
        {
            "model_names": [
                "pt_opt_facebook_opt_350m_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_350m_seq_cls_hf",
                "pt_bloom_bigscience_bloom_1b1_clm_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater10,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater11,
        [((1, 256, 6, 20), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater12,
        [((1, 256, 12, 40), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater13,
        [((1, 128, 12, 40), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater14,
        [((1, 128, 24, 80), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater15,
        [((1, 64, 24, 80), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater16,
        [((1, 64, 48, 160), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater17,
        [((1, 32, 48, 160), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater18,
        [((1, 32, 96, 320), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater19,
        [((1, 16, 96, 320), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater20,
        [((1, 16, 192, 640), torch.float32)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater10,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Greater10,
        [((2, 1, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Greater10,
        [((1, 12, 384, 384), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Greater21,
        [((1, 6, 2048), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99},
    ),
    (
        Greater22,
        [((1, 32, 480, 640), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater23,
        [((1, 64, 240, 320), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater24,
        [((1, 32, 240, 320), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater25,
        [((1, 128, 120, 160), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater26,
        [((1, 64, 120, 160), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater27,
        [((1, 256, 60, 80), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater28,
        [((1, 128, 60, 80), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater29,
        [((1, 512, 30, 40), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater30,
        [((1, 256, 30, 40), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater31,
        [((1, 1024, 15, 20), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater32,
        [((1, 512, 15, 20), torch.float32)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Greater33,
        [((1, 6, 3072), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_790m_hf_clm_hf"], "pcc": 0.99},
    ),
    (Greater10, [((1, 1, 7, 7), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Greater10,
        [((2, 1, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Greater")

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
