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


class Matmul0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul0.weight_1",
            forge.Parameter(*(768, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul0.weight_1"))
        return matmul_output_1


class Matmul1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, matmul_input_0, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, matmul_input_1)
        return matmul_output_1


class Matmul2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul2.weight_1",
            forge.Parameter(*(768, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul2.weight_1"))
        return matmul_output_1


class Matmul3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul3.weight_1",
            forge.Parameter(*(3072, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul3.weight_1"))
        return matmul_output_1


class Matmul4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul4.weight_1",
            forge.Parameter(*(1024, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul4.weight_1"))
        return matmul_output_1


class Matmul5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul5.weight_1",
            forge.Parameter(*(1024, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul5.weight_1"))
        return matmul_output_1


class Matmul6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul6.weight_1",
            forge.Parameter(*(4096, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul6.weight_1"))
        return matmul_output_1


class Matmul7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul7.weight_1",
            forge.Parameter(*(768, 30522), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul7.weight_1"))
        return matmul_output_1


class Matmul8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul8.weight_1",
            forge.Parameter(*(384, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul8.weight_1"))
        return matmul_output_1


class Matmul9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul9.weight_1",
            forge.Parameter(*(384, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul9.weight_1"))
        return matmul_output_1


class Matmul10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul10.weight_1",
            forge.Parameter(*(1536, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul10.weight_1"))
        return matmul_output_1


class Matmul11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul11.weight_1",
            forge.Parameter(*(256, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul11.weight_1"))
        return matmul_output_1


class Matmul12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul12.weight_1",
            forge.Parameter(*(256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul12.weight_1"))
        return matmul_output_1


class Matmul13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul13.weight_1",
            forge.Parameter(*(2048, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul13.weight_1"))
        return matmul_output_1


class Matmul14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul14.weight_1",
            forge.Parameter(*(256, 92), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul14.weight_1"))
        return matmul_output_1


class Matmul15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul15.weight_1",
            forge.Parameter(*(256, 4), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul15.weight_1"))
        return matmul_output_1


class Matmul16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul16.weight_1",
            forge.Parameter(*(256, 251), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul16.weight_1"))
        return matmul_output_1


class Matmul17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul17.weight_1",
            forge.Parameter(*(9216, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul17.weight_1"))
        return matmul_output_1


class Matmul18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul18.weight_1",
            forge.Parameter(*(4096, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul18.weight_1"))
        return matmul_output_1


class Matmul19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul19.weight_1",
            forge.Parameter(*(4096, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul19.weight_1"))
        return matmul_output_1


class Matmul20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul20.weight_1",
            forge.Parameter(*(1024, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul20.weight_1"))
        return matmul_output_1


class Matmul21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul21.weight_1",
            forge.Parameter(*(1152, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul21.weight_1"))
        return matmul_output_1


class Matmul22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul22.weight_1",
            forge.Parameter(*(1280, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul22.weight_1"))
        return matmul_output_1


class Matmul23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul23.weight_1",
            forge.Parameter(*(512, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul23.weight_1"))
        return matmul_output_1


class Matmul24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul24.weight_1",
            forge.Parameter(*(2048, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul24.weight_1"))
        return matmul_output_1


class Matmul25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul25_const_1", shape=(1, 1, 588), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul25_const_1"))
        return matmul_output_1


class Matmul26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul26_const_1", shape=(1, 1, 39), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul26_const_1"))
        return matmul_output_1


class Matmul27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul27_const_1", shape=(1, 1, 596), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul27_const_1"))
        return matmul_output_1


class Matmul28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul28_const_1", shape=(1, 1, 10), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul28_const_1"))
        return matmul_output_1


class Matmul29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul29_const_1", shape=(1, 1, 6), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul29_const_1"))
        return matmul_output_1


class Matmul30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul30_const_1", shape=(1, 1, 334), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul30_const_1"))
        return matmul_output_1


class Matmul31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul31_const_1", shape=(1, 1, 207), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul31_const_1"))
        return matmul_output_1


class Matmul32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul32_const_1", shape=(1, 1, 7), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul32_const_1"))
        return matmul_output_1


class Matmul33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul33_const_1", shape=(1, 1, 107), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul33_const_1"))
        return matmul_output_1


class Matmul34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul34_const_1", shape=(1, 1, 256), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul34_const_1"))
        return matmul_output_1


class Matmul35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul35_const_1", shape=(1, 1, 4), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul35_const_1"))
        return matmul_output_1


class Matmul36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul36_const_1", shape=(1, 1, 32), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul36_const_1"))
        return matmul_output_1


class Matmul37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul37_const_1", shape=(1, 1, 128), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul37_const_1"))
        return matmul_output_1


class Matmul38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul38_const_1", shape=(1, 1, 11), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul38_const_1"))
        return matmul_output_1


class Matmul39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul39_const_1", shape=(1, 1, 12), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul39_const_1"))
        return matmul_output_1


class Matmul40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul40_const_1", shape=(1, 1, 5), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul40_const_1"))
        return matmul_output_1


class Matmul41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul41_const_1", shape=(1, 1, 13), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul41_const_1"))
        return matmul_output_1


class Matmul42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul42_const_1", shape=(1, 1, 29), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul42_const_1"))
        return matmul_output_1


class Matmul43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul43_const_1", shape=(1, 1, 35), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul43_const_1"))
        return matmul_output_1


class Matmul44(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul44_const_1", shape=(4, 24), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul44_const_1"))
        return matmul_output_1


class Matmul45(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul45_const_1", shape=(4, 72), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul45_const_1"))
        return matmul_output_1


class Matmul46(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul46_const_1", shape=(12, 24), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul46_const_1"))
        return matmul_output_1


class Matmul47(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul47_const_1", shape=(12, 72), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul47_const_1"))
        return matmul_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Matmul0,
        [((6, 768), torch.float32)],
        {
            "model_name": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 6, 64), torch.float32), ((12, 64, 6), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 6, 6), torch.float32), ((12, 6, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 6, 768), torch.float32)],
        {
            "model_name": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul3,
        [((1, 6, 3072), torch.float32)],
        {
            "model_name": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_vilt_dandelin_vilt_b32_mlm_mlm_hf",
                "pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf",
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul4,
        [((384, 1024), torch.float32)],
        {"model_name": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 384, 64), torch.float32), ((16, 64, 384), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 384, 384), torch.float32), ((16, 384, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul5,
        [((1, 384, 1024), torch.float32)],
        {"model_name": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul6,
        [((1, 384, 4096), torch.float32)],
        {"model_name": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((384, 1024), torch.float32), ((1024, 1), torch.float32)],
        {
            "model_name": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Matmul0, [((128, 768), torch.float32)], {"model_name": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((12, 128, 64), torch.float32), ((12, 64, 128), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 128, 128), torch.float32), ((12, 128, 64), torch.float32)],
        {
            "model_name": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Matmul2, [((1, 128, 768), torch.float32)], {"model_name": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Matmul3, [((1, 128, 3072), torch.float32)], {"model_name": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Matmul0, [((1, 128, 768), torch.float32)], {"model_name": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (Matmul7, [((1, 128, 768), torch.float32)], {"model_name": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (
        Matmul8,
        [((13, 384), torch.float32)],
        {"model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 13, 32), torch.float32), ((12, 32, 13), torch.float32)],
        {"model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 13, 13), torch.float32), ((12, 13, 32), torch.float32)],
        {"model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul9,
        [((1, 13, 384), torch.float32)],
        {"model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul10,
        [((1, 13, 1536), torch.float32)],
        {"model_name": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384), torch.float32), ((384, 384), torch.float32)],
        {
            "model_name": [
                "onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf",
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul11,
        [((100, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 100, 32), torch.float32), ((8, 32, 100), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((100, 256), torch.float32), ((256, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 100, 100), torch.float32), ((8, 100, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul11,
        [((1, 100, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul11,
        [((280, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 280, 32), torch.float32), ((8, 32, 280), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul11,
        [((1, 280, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 280, 280), torch.float32), ((8, 280, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul12,
        [((1, 280, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul13,
        [((1, 280, 2048), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 100, 32), torch.float32), ((8, 32, 280), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 100, 280), torch.float32), ((8, 280, 32), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul12,
        [((1, 100, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul13,
        [((1, 100, 2048), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul14,
        [((100, 256), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_obj_det_hf"], "pcc": 0.99},
    ),
    (
        Matmul15,
        [((1, 100, 256), torch.float32)],
        {
            "model_name": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul16,
        [((100, 256), torch.float32)],
        {"model_name": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2048), torch.float32), ((2048, 1000), torch.float32)],
        {
            "model_name": [
                "onnx_resnet_50_img_cls_hf",
                "pt_hrnet_hrnetv2_w64_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w48_pose_estimation_osmr",
                "pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w32_pose_estimation_osmr",
                "pt_hrnet_hrnet_w40_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w40_pose_estimation_osmr",
                "pt_hrnet_hrnetv2_w44_pose_estimation_osmr",
                "pt_hrnet_hrnet_w30_pose_estimation_timm",
                "pt_hrnet_hrnet_w48_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr",
                "pt_hrnet_hrnet_w44_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w30_pose_estimation_osmr",
                "pt_hrnet_hrnet_w64_pose_estimation_timm",
                "pt_hrnet_hrnet_w18_small_pose_estimation_timm",
                "pt_hrnet_hrnetv2_w18_pose_estimation_osmr",
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
                "pt_resnext_resnext101_64x4d_img_cls_osmr",
                "pt_resnext_resnext50_32x4d_img_cls_osmr",
                "pt_resnext_resnext14_32x4d_img_cls_osmr",
                "pt_resnext_resnext26_32x4d_img_cls_osmr",
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
        },
    ),
    (
        Matmul0,
        [((197, 768), torch.float32)],
        {"model_name": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 197, 64), torch.float32), ((12, 64, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 197, 197), torch.float32), ((12, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 197, 768), torch.float32)],
        {"model_name": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul3,
        [((1, 197, 3072), torch.float32)],
        {"model_name": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 1000), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul4,
        [((197, 1024), torch.float32)],
        {"model_name": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 197, 64), torch.float32), ((16, 64, 197), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 197, 197), torch.float32), ((16, 197, 64), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul5,
        [((1, 197, 1024), torch.float32)],
        {"model_name": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul6,
        [((1, 197, 4096), torch.float32)],
        {"model_name": ["onnx_vit_base_google_vit_large_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024), torch.float32), ((1024, 1000), torch.float32)],
        {
            "model_name": [
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_densenet_densenet121_img_cls_torchvision",
                "pt_googlenet_base_img_cls_torchvision",
                "pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_l16_224_img_cls_timm",
                "pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm",
                "pt_swin_swin_b_img_cls_torchvision",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
                "pt_vovnet_vovnet57_obj_det_osmr",
                "pt_vovnet_vovnet39_obj_det_osmr",
                "pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub",
                "pt_vovnet_vovnet_v1_57_obj_det_torchhub",
                "pt_vovnet_v1_vovnet39_obj_det_torchhub",
                "pt_vovnet_ese_vovnet39b_obj_det_torchhub",
                "pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (Matmul17, [((1, 9216), torch.float32)], {"model_name": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (Matmul18, [((1, 4096), torch.float32)], {"model_name": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (Matmul19, [((1, 4096), torch.float32)], {"model_name": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Matmul20,
        [((1, 1024), torch.float32)],
        {
            "model_name": [
                "pd_densenet_121_img_cls_paddlemodels",
                "pd_googlenet_base_img_cls_paddlemodels",
                "pd_mobilenetv1_basic_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (Matmul21, [((1, 1152), torch.float32)], {"model_name": ["pd_googlenet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Matmul22,
        [((1, 1280), torch.float32)],
        {"model_name": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul23,
        [((1, 512), torch.float32)],
        {"model_name": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul24,
        [((1, 2048), torch.float32)],
        {
            "model_name": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "pd_resnet_50_img_cls_paddlemodels",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((2, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1, 64), torch.float32), ((64, 64, 1), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1, 1), torch.float32), ((64, 1, 64), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 13, 64), torch.float32), ((24, 64, 13), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 13, 13), torch.float32), ((24, 13, 64), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((26, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_name": [
                "pt_stereo_facebook_musicgen_large_music_generation_hf",
                "pt_stereo_facebook_musicgen_small_music_generation_hf",
                "pt_stereo_facebook_musicgen_medium_music_generation_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1, 64), torch.float32), ((64, 64, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 1, 13), torch.float32), ((64, 13, 64), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_large_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1, 64), torch.float32), ((32, 64, 1), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1, 1), torch.float32), ((32, 1, 64), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 1024), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1, 64), torch.float32), ((32, 64, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1, 13), torch.float32), ((32, 13, 64), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1024), torch.float32), ((1024, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_small_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 1, 64), torch.float32), ((48, 64, 1), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 1, 1), torch.float32), ((48, 1, 64), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 768), torch.float32), ((768, 1536), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((26, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 1, 64), torch.float32), ((48, 64, 13), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 1, 13), torch.float32), ((48, 13, 64), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1536), torch.float32), ((1536, 6144), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 6144), torch.float32), ((6144, 1536), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1536), torch.float32), ((1536, 2048), torch.float32)],
        {"model_name": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 64), torch.float32), ((6, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((6, 1, 1), torch.float32), ((6, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_tiny_speech_recognition_hf",
                "pt_t5_google_flan_t5_small_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 384), torch.float32), ((384, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 384), torch.float32), ((384, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1500, 64), torch.float32), ((6, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1500, 1500), torch.float32), ((6, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 384), torch.float32), ((384, 1536), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 1536), torch.float32), ((1536, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 64), torch.float32), ((6, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 1500), torch.float32), ((6, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 384), torch.float32), ((384, 1536), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1536), torch.float32), ((1536, 384), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 384), torch.float32), ((384, 51865), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 1, 64), torch.float32), ((20, 64, 1), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 1, 1), torch.float32), ((20, 1, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_large_speech_recognition_hf",
                "pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((20, 1500, 64), torch.float32), ((20, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 1500, 1500), torch.float32), ((20, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 1280), torch.float32), ((1280, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 5120), torch.float32), ((5120, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 1, 64), torch.float32), ((20, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 1, 1500), torch.float32), ((20, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1280), torch.float32), ((1280, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 5120), torch.float32), ((5120, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1280), torch.float32), ((1280, 51865), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 64), torch.float32), ((12, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 1, 1), torch.float32), ((12, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_small_speech_recognition_hf",
                "pt_t5_google_flan_t5_base_text_gen_hf",
                "pt_t5_t5_base_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1500, 64), torch.float32), ((12, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1500, 1500), torch.float32), ((12, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 64), torch.float32), ((12, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 1500), torch.float32), ((12, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf", "pt_t5_t5_base_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 768), torch.float32), ((768, 51865), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 1, 64), torch.float32), ((16, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 1, 1), torch.float32), ((16, 1, 64), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_t5_t5_large_text_gen_hf",
                "pt_t5_google_flan_t5_large_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_whisper_openai_whisper_medium_speech_recognition_hf",
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1500, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 1500, 64), torch.float32), ((16, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 1500, 1500), torch.float32), ((16, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 1, 64), torch.float32), ((16, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 1, 1500), torch.float32), ((16, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf", "pt_t5_t5_large_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 1024), torch.float32), ((1024, 51865), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_medium_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512), torch.float32), ((512, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 1, 64), torch.float32), ((8, 64, 1), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 1, 1), torch.float32), ((8, 1, 64), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1500, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1500, 64), torch.float32), ((8, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1500, 1500), torch.float32), ((8, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1500, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1, 64), torch.float32), ((8, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1, 1500), torch.float32), ((8, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((512, 2048), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 2048), torch.float32), ((2048, 512), torch.float32)],
        {
            "model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf", "pt_t5_t5_small_text_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((512, 51865), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 2, 64), torch.float32), ((20, 64, 2), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 2, 2), torch.float32), ((20, 2, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 2, 64), torch.float32), ((20, 64, 1500), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 2, 1500), torch.float32), ((20, 1500, 64), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2, 1280), torch.float32), ((1280, 5120), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2, 5120), torch.float32), ((5120, 1280), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2, 1280), torch.float32), ((1280, 51866), torch.float32)],
        {"model_name": ["pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 7, 64), torch.float32), ((16, 64, 7), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 7, 7), torch.float32), ((16, 7, 64), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_name": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((588, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul25,
        [((1, 64, 1), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 588, 128), torch.float32), ((16, 128, 588), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 588, 588), torch.float32), ((16, 588, 128), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((588, 2048), torch.float32), ((2048, 5504), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 588, 5504), torch.float32), ((5504, 2048), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 588, 2048), torch.float32), ((2048, 32256), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul26,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_deepseek_deepseek_math_7b_instruct_qa_hf",
                "pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 39, 128), torch.float32), ((32, 128, 39), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 39, 39), torch.float32), ((32, 39, 128), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 4096), torch.float32), ((4096, 11008), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 39, 11008), torch.float32), ((11008, 4096), torch.float32)],
            {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 39, 4096), torch.float32), ((4096, 102400), torch.float32)],
        {"model_name": ["pt_deepseek_deepseek_math_7b_instruct_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((577, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 577, 64), torch.float32), ((16, 64, 577), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 577, 577), torch.float32), ((16, 577, 64), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 577, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 577, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 576, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 576, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((596, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul27,
        [((1, 64, 1), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 596, 128), torch.float32), ((32, 128, 596), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 596, 596), torch.float32), ((32, 596, 128), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((596, 4096), torch.float32), ((4096, 11008), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 596, 11008), torch.float32), ((11008, 4096), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 596, 4096), torch.float32), ((4096, 32064), torch.float32)],
        {"model_name": ["pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((204, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 204, 64), torch.float32), ((12, 64, 204), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 204, 204), torch.float32), ((12, 204, 64), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 204, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 204, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((201, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 201, 64), torch.float32), ((12, 64, 201), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 201, 201), torch.float32), ((12, 201, 64), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 201, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 201, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 1536), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1536), torch.float32), ((1536, 3129), torch.float32)],
        {"model_name": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 128, 128), torch.float32), ((16, 128, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xlarge_v1_token_cls_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 128), torch.float32)],
        {"model_name": ["pt_albert_xlarge_v1_mlm_hf", "pt_albert_xlarge_v2_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 30000), torch.float32)],
        {
            "model_name": [
                "pt_albert_xlarge_v1_mlm_hf",
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_xlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_roberta_xlm_roberta_base_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_name": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_base_v1_mlm_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_bert_bert_base_uncased_mlm_hf",
                "pt_distilbert_distilbert_base_multilingual_cased_mlm_hf",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_roberta_xlm_roberta_base_mlm_hf",
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_name": ["pt_albert_base_v1_token_cls_hf", "pt_albert_base_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 128, 64), torch.float32), ((16, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 128, 128), torch.float32), ((16, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_albert_large_v2_mlm_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_large_v1_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 128), torch.float32)],
        {"model_name": ["pt_albert_large_v2_mlm_hf", "pt_albert_large_v1_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 14, 128), torch.float32), ((128, 768), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 14, 64), torch.float32), ((12, 64, 14), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 14, 14), torch.float32), ((12, 14, 64), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 14, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 14, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 14, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 768), torch.float32), ((768, 1), torch.float32)],
        {"model_name": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 128), torch.float32)],
        {"model_name": ["pt_albert_base_v1_mlm_hf", "pt_albert_base_v2_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 9, 128), torch.float32), ((128, 768), torch.float32)],
        {"model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((9, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 9, 64), torch.float32), ((12, 64, 9), torch.float32)],
        {"model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 9, 9), torch.float32), ((12, 9, 64), torch.float32)],
        {"model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 9, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 9, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 9, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 2), torch.float32)],
        {
            "model_name": [
                "pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf",
                "pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf",
                "pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 128), torch.float32), ((128, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((128, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_mistral_mistralai_mistral_7b_v0_1_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 128, 64), torch.float32), ((64, 64, 128), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((64, 128, 128), torch.float32), ((64, 128, 64), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 16384), torch.float32)],
        {
            "model_name": [
                "pt_albert_xxlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 128, 16384), torch.float32), ((16384, 4096), torch.float32)],
            {
                "model_name": [
                    "pt_albert_xxlarge_v2_token_cls_hf",
                    "pt_albert_xxlarge_v1_token_cls_hf",
                    "pt_albert_xxlarge_v1_mlm_hf",
                    "pt_albert_xxlarge_v2_mlm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 2), torch.float32)],
        {"model_name": ["pt_albert_xxlarge_v2_token_cls_hf", "pt_albert_xxlarge_v1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 2), torch.float32)],
        {"model_name": ["pt_albert_large_v2_token_cls_hf", "pt_albert_large_v1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_name": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 128), torch.float32)],
        {"model_name": ["pt_albert_xxlarge_v1_mlm_hf", "pt_albert_xxlarge_v2_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 256, 64), torch.float32), ((16, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 256, 256), torch.float32), ((16, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_name": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf", "pt_xglm_facebook_xglm_564m_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((384, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 384, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 384, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 128, 1024), torch.float32), ((1024, 9), torch.float32)],
        {"model_name": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 30522), torch.float32)],
        {
            "model_name": ["pt_bert_bert_base_uncased_mlm_hf", "pt_distilbert_distilbert_base_uncased_mlm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((6, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 6, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 6, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 1536), torch.float32), ((1536, 4608), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 96), torch.float32), ((16, 96, 32), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 32), torch.float32), ((16, 32, 96), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 1536), torch.float32), ((1536, 6144), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 6144), torch.float32), ((6144, 1536), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 1536), torch.float32), ((1536, 250880), torch.float32)],
        {"model_name": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 51200), torch.float32)],
        {
            "model_name": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_codegen_salesforce_codegen_350m_mono_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((384, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 384, 64), torch.float32), ((12, 64, 384), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 384, 384), torch.float32), ((12, 384, 64), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((384, 768), torch.float32), ((768, 1), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 119547), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_multilingual_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 9), torch.float32)],
        {"model_name": ["pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 28996), torch.float32)],
        {"model_name": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 768), torch.float32), ((768, 1), torch.float32)],
        {
            "model_name": [
                "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 768), torch.float32), ((768, 1), torch.float32)],
            {
                "model_name": [
                    "pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader",
                    "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((10, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul28,
        [((1, 128, 1), torch.float32)],
        {
            "model_name": [
                "pt_falcon3_tiiuae_falcon3_1b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_3b_base_clm_hf",
                "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((10, 2048), torch.float32), ((2048, 1024), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 10, 256), torch.float32), ((8, 256, 10), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 10, 10), torch.float32), ((8, 10, 256), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((10, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 10, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 10, 2048), torch.float32), ((2048, 131072), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_1b_base_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 4544), torch.float32), ((4544, 18176), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 6, 18176), torch.float32), ((18176, 4544), torch.float32)],
            {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((6, 4544), torch.float32), ((4544, 4672), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul29,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf", "pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((71, 6, 64), torch.float32), ((1, 64, 6), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((71, 6, 6), torch.float32), ((1, 6, 64), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 4544), torch.float32), ((4544, 4544), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 6, 4544), torch.float32), ((4544, 65024), torch.float32)],
        {"model_name": ["pt_falcon_tiiuae_falcon_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((10, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((10, 3072), torch.float32), ((3072, 1024), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 10, 256), torch.float32), ((12, 256, 10), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 10, 10), torch.float32), ((12, 10, 256), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((10, 3072), torch.float32), ((3072, 9216), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 10, 9216), torch.float32), ((9216, 3072), torch.float32)],
            {"model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 10, 3072), torch.float32), ((3072, 131072), torch.float32)],
        {
            "model_name": ["pt_falcon3_tiiuae_falcon3_3b_base_clm_hf", "pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((10, 3072), torch.float32), ((3072, 23040), torch.float32)],
        {"model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 10, 23040), torch.float32), ((23040, 3072), torch.float32)],
            {"model_name": ["pt_falcon3_tiiuae_falcon3_7b_base_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 334, 4096), torch.float32), ((4096, 12288), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (Matmul30, [((1, 16, 1), torch.float32)], {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((64, 334, 64), torch.float32), ((64, 64, 334), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 334, 334), torch.float32), ((64, 334, 64), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((334, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 334, 4096), torch.float32), ((4096, 16384), torch.float32)],
        {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 334, 16384), torch.float32), ((16384, 4096), torch.float32)],
            {"model_name": ["pt_fuyu_adept_fuyu_8b_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((207, 3584), torch.float32), ((3584, 4096), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul31,
        [((1, 128, 1), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf", "pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((207, 3584), torch.float32), ((3584, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 207, 256), torch.float32), ((16, 256, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 207, 207), torch.float32), ((16, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((207, 4096), torch.float32), ((4096, 3584), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((207, 3584), torch.float32), ((3584, 14336), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 207, 14336), torch.float32), ((14336, 3584), torch.float32)],
            {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 207, 3584), torch.float32), ((3584, 256000), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_9b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((7, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (Matmul32, [((1, 128, 1), torch.float32)], {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((7, 2048), torch.float32), ((2048, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 7, 256), torch.float32), ((8, 256, 7), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 7, 7), torch.float32), ((8, 7, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((7, 2048), torch.float32), ((2048, 16384), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 7, 16384), torch.float32), ((16384, 2048), torch.float32)],
            {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 7, 2048), torch.float32), ((2048, 256000), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2b_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((207, 2304), torch.float32), ((2304, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((207, 2304), torch.float32), ((2304, 1024), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 207, 256), torch.float32), ((8, 256, 207), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 207, 207), torch.float32), ((8, 207, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((207, 2048), torch.float32), ((2048, 2304), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((207, 2304), torch.float32), ((2304, 9216), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 207, 9216), torch.float32), ((9216, 2304), torch.float32)],
            {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 207, 2304), torch.float32), ((2304, 256000), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_2_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul33,
        [((1, 128, 1), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf", "pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 2048), torch.float32), ((2048, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 107, 256), torch.float32), ((8, 256, 107), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 107, 107), torch.float32), ((8, 107, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 2048), torch.float32), ((2048, 16384), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 107, 16384), torch.float32), ((16384, 2048), torch.float32)],
            {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 107, 2048), torch.float32), ((2048, 256000), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_2b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 3072), torch.float32), ((3072, 4096), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 107, 256), torch.float32), ((16, 256, 107), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 107, 107), torch.float32), ((16, 107, 256), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 4096), torch.float32), ((4096, 3072), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((107, 3072), torch.float32), ((3072, 24576), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 107, 24576), torch.float32), ((24576, 3072), torch.float32)],
            {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 107, 3072), torch.float32), ((3072, 256000), torch.float32)],
        {"model_name": ["pt_gemma_google_gemma_1_1_7b_it_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((7, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 7, 64), torch.float32), ((12, 64, 7), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 7, 7), torch.float32), ((12, 7, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul0,
        [((7, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((7, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul3,
        [((7, 3072), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((7, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_name": ["pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 256, 64), torch.float32), ((12, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 256, 256), torch.float32), ((12, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_gpt2_gpt2_text_gen_hf",
                "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf",
                "pt_opt_facebook_opt_125m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Matmul0, [((256, 768), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Matmul2, [((256, 768), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Matmul3, [((256, 3072), torch.float32)], {"model_name": ["pt_gpt2_gpt2_text_gen_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((1, 256, 768), torch.float32), ((768, 50257), torch.float32)],
        {"model_name": ["pt_gpt2_gpt2_text_gen_hf", "pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf",
                "pt_opt_facebook_opt_1_3b_qa_hf",
                "pt_opt_facebook_opt_1_3b_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 32, 128), torch.float32), ((16, 128, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 32), torch.float32), ((16, 32, 128), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                "pt_phi2_microsoft_phi_2_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((20, 256, 128), torch.float32), ((20, 128, 256), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 256, 256), torch.float32), ((20, 256, 128), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 256, 10240), torch.float32), ((10240, 2560), torch.float32)],
            {
                "model_name": [
                    "pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf",
                    "pt_phi2_microsoft_phi_2_pytdml_clm_hf",
                    "pt_phi2_microsoft_phi_2_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 256, 2560), torch.float32), ((2560, 50257), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 32, 64), torch.float32), ((12, 64, 32), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 32, 32), torch.float32), ((12, 32, 64), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf",
                "pt_opt_facebook_opt_125m_qa_hf",
                "pt_opt_facebook_opt_125m_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 32, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 32, 128), torch.float32), ((20, 128, 32), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((20, 32, 32), torch.float32), ((20, 32, 128), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 32, 10240), torch.float32), ((10240, 2560), torch.float32)],
            {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 32, 2560), torch.float32), ((2560, 2), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 256, 128), torch.float32), ((16, 128, 256), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 256, 256), torch.float32), ((16, 256, 128), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf", "pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {
            "model_name": [
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_xglm_facebook_xglm_1_7b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 50257), torch.float32)],
        {"model_name": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                "pt_opt_facebook_opt_350m_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 256, 128), torch.float32), ((32, 128, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 256, 256), torch.float32), ((32, 256, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 4096), torch.float32), ((4096, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 256, 14336), torch.float32), ((14336, 4096), torch.float32)],
            {
                "model_name": [
                    "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                    "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                    "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                    "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 256, 4096), torch.float32), ((4096, 128256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_1_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_clm_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 2048), torch.float32), ((2048, 512), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 256, 64), torch.float32), ((32, 64, 256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 256, 256), torch.float32), ((32, 256, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
                "pt_opt_facebook_opt_1_3b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 128256), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul35,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 4, 128), torch.float32), ((32, 128, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 4, 4), torch.float32), ((32, 4, 128), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_huggyllama_llama_7b_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 14336), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4, 14336), torch.float32), ((14336, 4096), torch.float32)],
            {
                "model_name": [
                    "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                    "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                    "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                    "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 4, 4096), torch.float32), ((4096, 2), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf",
                "pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 11008), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4, 11008), torch.float32), ((11008, 4096), torch.float32)],
            {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((4, 4096), torch.float32), ((4096, 2), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4, 3072), torch.float32), ((3072, 1024), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((24, 4, 128), torch.float32), ((24, 128, 4), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((24, 4, 4), torch.float32), ((24, 4, 128), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4, 8192), torch.float32), ((8192, 3072), torch.float32)],
            {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 4, 3072), torch.float32), ((3072, 2), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul35,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 2048), torch.float32), ((2048, 512), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 4, 64), torch.float32), ((32, 64, 4), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 4, 4), torch.float32), ((32, 4, 64), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 4, 8192), torch.float32), ((8192, 2048), torch.float32)],
            {
                "model_name": [
                    "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                    "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 4, 2048), torch.float32), ((2048, 2), torch.float32)],
        {
            "model_name": [
                "pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf",
                "pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul36,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf", "pt_llama3_huggyllama_llama_7b_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 3072), torch.float32), ((3072, 1024), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((24, 32, 128), torch.float32), ((24, 128, 32), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((24, 32, 32), torch.float32), ((24, 32, 128), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 3072), torch.float32), ((3072, 128256), torch.float32)],
        {"model_name": ["pt_llama3_meta_llama_llama_3_2_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 32, 128), torch.float32), ((32, 128, 32), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 32, 32), torch.float32), ((32, 32, 128), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 4096), torch.float32), ((4096, 11008), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 32, 11008), torch.float32), ((11008, 4096), torch.float32)],
            {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 32, 4096), torch.float32), ((4096, 32000), torch.float32)],
        {"model_name": ["pt_llama3_huggyllama_llama_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul37,
        [((1, 64, 1), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 128, 128), torch.float32), ((32, 128, 128), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 4096), torch.float32), ((4096, 14336), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 128, 14336), torch.float32), ((14336, 4096), torch.float32)],
            {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 128, 4096), torch.float32), ((4096, 32000), torch.float32)],
        {"model_name": ["pt_mistral_mistralai_mistral_7b_v0_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 32, 64), torch.float32), ((32, 64, 32), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 32, 32), torch.float32), ((32, 32, 64), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_qa_hf", "pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 2048), torch.float32), ((2048, 1), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 32, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 64), torch.float32), ((16, 64, 32), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 32, 32), torch.float32), ((16, 32, 64), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf", "pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 512), torch.float32), ((512, 2), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 50272), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 512), torch.float32), ((512, 1), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 512), torch.float32), ((512, 50272), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_350m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 768), torch.float32), ((768, 1), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 768), torch.float32), ((768, 50272), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_name": ["pt_opt_facebook_opt_1_3b_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 50176, 256), torch.float32), ((256, 256), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((50176, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512, 512), torch.float32), ((1, 512, 50176), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 50176), torch.float32), ((1, 50176, 512), torch.float32)],
            {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 512, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((512, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 512, 128), torch.float32), ((8, 128, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 512, 512), torch.float32), ((8, 512, 128), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 1024), torch.float32), ((1, 1024, 512), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((1, 512, 1024), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1, 1024), torch.float32), ((1024, 1000), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 512, 1024), torch.float32), ((1024, 322), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3025, 322), torch.float32), ((322, 322), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512, 322), torch.float32), ((1, 322, 3025), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512, 3025), torch.float32), ((1, 3025, 322), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512, 322), torch.float32), ((322, 1024), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512, 1024), torch.float32), ((1024, 261), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((50176, 261), torch.float32), ((261, 261), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512, 261), torch.float32), ((1, 261, 50176), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 512, 50176), torch.float32), ((1, 50176, 261), torch.float32)],
            {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 512, 261), torch.float32), ((261, 1024), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2048, 768), torch.float32), ((768, 256), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 1280), torch.float32), ((1280, 256), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2048, 768), torch.float32), ((768, 256), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 256, 32), torch.float32), ((8, 32, 2048), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2048, 768), torch.float32), ((768, 1280), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 256, 2048), torch.float32), ((8, 2048, 160), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 1280), torch.float32), ((1280, 1280), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1280), torch.float32), ((1280, 256), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 256, 32), torch.float32), ((8, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_perceiverio_deepmind_language_perceiver_mlm_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 256, 256), torch.float32), ((8, 256, 160), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 2048, 32), torch.float32), ((8, 32, 256), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 1280), torch.float32), ((1280, 768), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 2048, 256), torch.float32), ((8, 256, 96), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2048, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2048, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2048, 768), torch.float32), ((768, 262), torch.float32)],
        {"model_name": ["pt_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul34,
        [((1, 16, 1), torch.float32)],
        {"model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 256, 80), torch.float32), ((32, 80, 256), torch.float32)],
        {"model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 256, 256), torch.float32), ((32, 256, 80), torch.float32)],
        {"model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {"model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2560), torch.float32), ((2560, 51200), torch.float32)],
        {"model_name": ["pt_phi2_microsoft_phi_2_pytdml_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((11, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul38,
        [((1, 16, 1), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 11, 80), torch.float32), ((32, 80, 11), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 11, 11), torch.float32), ((32, 11, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((11, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 11, 10240), torch.float32), ((10240, 2560), torch.float32)],
            {
                "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 11, 2560), torch.float32), ((2560, 2), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf", "pt_phi2_microsoft_phi_2_seq_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul39,
        [((1, 16, 1), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 12, 80), torch.float32), ((32, 80, 12), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 12, 12), torch.float32), ((32, 12, 80), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 12, 10240), torch.float32), ((10240, 2560), torch.float32)],
            {
                "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 12, 2560), torch.float32), ((2560, 2), torch.float32)],
        {
            "model_name": ["pt_phi2_microsoft_phi_2_pytdml_token_cls_hf", "pt_phi2_microsoft_phi_2_token_cls_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 5, 3072), torch.float32), ((3072, 9216), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul40,
        [((1, 48, 1), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 5, 96), torch.float32), ((32, 96, 5), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 5, 5), torch.float32), ((32, 5, 96), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((5, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((5, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 5, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 5, 3072), torch.float32), ((3072, 2), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 13, 3072), torch.float32), ((3072, 9216), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul41,
        [((1, 48, 1), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 13, 96), torch.float32), ((32, 96, 13), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 13, 13), torch.float32), ((32, 13, 96), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((13, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((13, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 13, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 13, 3072), torch.float32), ((3072, 2), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 3072), torch.float32), ((3072, 9216), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul34,
        [((1, 48, 1), torch.float32)],
        {"model_name": ["pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 256, 96), torch.float32), ((32, 96, 256), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((32, 256, 256), torch.float32), ((32, 256, 96), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 3072), torch.float32), ((3072, 3072), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 3072), torch.float32), ((3072, 8192), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 8192), torch.float32), ((8192, 3072), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 3072), torch.float32), ((3072, 32064), torch.float32)],
        {
            "model_name": [
                "pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf",
                "pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((29, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul42,
        [((1, 32, 1), torch.float32)],
        {
            "model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 29, 64), torch.float32), ((16, 64, 29), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 29, 29), torch.float32), ((16, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 29, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 29, 1024), torch.float32), ((1024, 151936), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 6, 64), torch.float32), ((16, 64, 6), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 6, 6), torch.float32), ((16, 6, 64), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 6, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 6, 1024), torch.float32), ((1024, 151936), torch.float32)],
        {"model_name": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((35, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul43,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 1536), torch.float32), ((1536, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 35, 128), torch.float32), ((12, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((12, 35, 35), torch.float32), ((12, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 1536), torch.float32), ((1536, 8960), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 35, 8960), torch.float32), ((8960, 1536), torch.float32)],
            {
                "model_name": [
                    "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                    "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 35, 1536), torch.float32), ((1536, 151936), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 3584), torch.float32), ((3584, 512), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((28, 35, 128), torch.float32), ((28, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((28, 35, 35), torch.float32), ((28, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 35, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {
                "model_name": [
                    "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                    "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 35, 3584), torch.float32), ((3584, 152064), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 896), torch.float32), ((896, 896), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul43,
        [((1, 32, 1), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((35, 896), torch.float32), ((896, 128), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 35, 64), torch.float32), ((14, 64, 35), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 35, 35), torch.float32), ((14, 35, 64), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((35, 896), torch.float32), ((896, 4864), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 35, 4864), torch.float32), ((4864, 896), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 35, 896), torch.float32), ((896, 151936), torch.float32)],
        {"model_name": ["pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((35, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 2048), torch.float32), ((2048, 256), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 35, 128), torch.float32), ((16, 128, 35), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((16, 35, 35), torch.float32), ((16, 35, 128), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((35, 2048), torch.float32), ((2048, 11008), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 35, 11008), torch.float32), ((11008, 2048), torch.float32)],
            {
                "model_name": [
                    "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                    "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 35, 2048), torch.float32), ((2048, 151936), torch.float32)],
        {
            "model_name": [
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf",
                "pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((39, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 2048), torch.float32), ((2048, 256), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 39, 128), torch.float32), ((16, 128, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 39, 39), torch.float32), ((16, 39, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 2048), torch.float32), ((2048, 11008), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 39, 11008), torch.float32), ((11008, 2048), torch.float32)],
            {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 39, 2048), torch.float32), ((2048, 151936), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul42,
        [((1, 64, 1), torch.float32)],
        {
            "model_name": [
                "pt_qwen_v2_qwen_qwen2_5_3b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf",
                "pt_qwen_v2_qwen_qwen2_5_7b_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((29, 2048), torch.float32), ((2048, 256), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 29, 128), torch.float32), ((16, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 29, 29), torch.float32), ((16, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 2048), torch.float32), ((2048, 11008), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 29, 11008), torch.float32), ((11008, 2048), torch.float32)],
            {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 29, 2048), torch.float32), ((2048, 151936), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_3b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1536), torch.float32), ((1536, 256), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 29, 128), torch.float32), ((12, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 29, 29), torch.float32), ((12, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 1536), torch.float32), ((1536, 8960), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 29, 8960), torch.float32), ((8960, 1536), torch.float32)],
            {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 29, 1536), torch.float32), ((1536, 151936), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 1536), torch.float32), ((1536, 256), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 39, 128), torch.float32), ((12, 128, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 39, 39), torch.float32), ((12, 39, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 1536), torch.float32), ((1536, 8960), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 39, 8960), torch.float32), ((8960, 1536), torch.float32)],
            {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 39, 1536), torch.float32), ((1536, 151936), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 896), torch.float32), ((896, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul26,
        [((1, 32, 1), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 896), torch.float32), ((896, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 39, 64), torch.float32), ((14, 64, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 39, 39), torch.float32), ((14, 39, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 896), torch.float32), ((896, 4864), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 39, 4864), torch.float32), ((4864, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 39, 896), torch.float32), ((896, 151936), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 3584), torch.float32), ((3584, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 39, 128), torch.float32), ((28, 128, 39), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 39, 39), torch.float32), ((28, 39, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((39, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 39, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 39, 3584), torch.float32), ((3584, 152064), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 896), torch.float32), ((896, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 896), torch.float32), ((896, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 29, 64), torch.float32), ((14, 64, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((14, 29, 29), torch.float32), ((14, 29, 64), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 896), torch.float32), ((896, 4864), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 29, 4864), torch.float32), ((4864, 896), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 29, 896), torch.float32), ((896, 151936), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((13, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (Matmul41, [((1, 64, 1), torch.float32)], {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((13, 3584), torch.float32), ((3584, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 13, 128), torch.float32), ((28, 128, 13), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 13, 13), torch.float32), ((28, 13, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((13, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 13, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 13, 3584), torch.float32), ((3584, 2), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_7b_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 3584), torch.float32), ((3584, 3584), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 3584), torch.float32), ((3584, 512), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 29, 128), torch.float32), ((28, 128, 29), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((28, 29, 29), torch.float32), ((28, 29, 128), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((29, 3584), torch.float32), ((3584, 18944), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 29, 18944), torch.float32), ((18944, 3584), torch.float32)],
            {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 29, 3584), torch.float32), ((3584, 152064), torch.float32)],
        {"model_name": ["pt_qwen_v2_qwen_qwen2_5_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128, 768), torch.float32), ((768, 250002), torch.float32)],
        {"model_name": ["pt_roberta_xlm_roberta_base_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 3), torch.float32)],
        {
            "model_name": [
                "pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((61, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 61, 64), torch.float32), ((12, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 61, 61), torch.float32), ((12, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 768), torch.float32), ((768, 2048), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 2048), torch.float32), ((2048, 768), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 64), torch.float32), ((12, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((12, 1, 61), torch.float32), ((12, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 2048), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 2048), torch.float32), ((2048, 768), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 768), torch.float32), ((768, 32128), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_base_text_gen_hf", "pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 61, 64), torch.float32), ((8, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 61, 61), torch.float32), ((8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1, 64), torch.float32), ((8, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 1, 61), torch.float32), ((8, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 512), torch.float32), ((512, 32128), torch.float32)],
        {"model_name": ["pt_t5_t5_small_text_gen_hf", "pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 61, 64), torch.float32), ((16, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 61, 61), torch.float32), ((16, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_name": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_name": ["pt_t5_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 1, 64), torch.float32), ((16, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((16, 1, 61), torch.float32), ((16, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1024), torch.float32), ((1024, 32128), torch.float32)],
        {"model_name": ["pt_t5_t5_large_text_gen_hf", "pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_large_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512), torch.float32), ((512, 384), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384), torch.float32), ((384, 512), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 512), torch.float32), ((512, 384), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 61, 64), torch.float32), ((6, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 61, 61), torch.float32), ((6, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 384), torch.float32), ((384, 512), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((61, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 61, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 64), torch.float32), ((6, 64, 61), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 1, 61), torch.float32), ((6, 61, 64), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512), torch.float32), ((512, 1024), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_name": ["pt_t5_google_flan_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 256008), torch.float32)],
        {"model_name": ["pt_xglm_facebook_xglm_1_7b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 256008), torch.float32)],
        {"model_name": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 72), torch.float32), ((72, 512), torch.float32)],
        {"model_name": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_name": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 512), torch.float32), ((512, 96), torch.float32)],
        {"model_name": ["pt_nbeats_generic_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 72), torch.float32), ((72, 256), torch.float32)],
        {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 256), torch.float32), ((256, 256), torch.float32)],
        {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 256), torch.float32), ((256, 8), torch.float32)],
        {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99},
    ),
    (Matmul44, [((1024, 4), torch.float32)], {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    (Matmul45, [((1024, 4), torch.float32)], {"model_name": ["pt_nbeats_trend_basis_clm_hf"], "pcc": 0.99}),
    (
        Matmul1,
        [((1024, 72), torch.float32), ((72, 2048), torch.float32)],
        {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1024, 2048), torch.float32), ((2048, 48), torch.float32)],
        {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99},
    ),
    (Matmul46, [((1024, 12), torch.float32)], {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    (Matmul47, [((1024, 12), torch.float32)], {"model_name": ["pt_nbeats_seasionality_basis_clm_hf"], "pcc": 0.99}),
    pytest.param(
        (
            Matmul1,
            [((1, 9216), torch.float32), ((9216, 4096), torch.float32)],
            {
                "model_name": [
                    "pt_alexnet_alexnet_img_cls_torchhub",
                    "pt_alexnet_base_img_cls_osmr",
                    "pt_rcnn_base_obj_det_torchvision_rect_0",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 4096), torch.float32), ((4096, 4096), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_rcnn_base_obj_det_torchvision_rect_0",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 4096), torch.float32), ((4096, 1000), torch.float32)],
        {
            "model_name": [
                "pt_alexnet_alexnet_img_cls_torchhub",
                "pt_alexnet_base_img_cls_osmr",
                "pt_vgg_vgg13_img_cls_torchvision",
                "pt_vgg_vgg19_img_cls_torchvision",
                "pt_vgg_vgg19_obj_det_osmr",
                "pt_vgg_vgg11_obj_det_osmr",
                "pt_vgg_bn_vgg19_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_torchhub",
                "pt_vgg_vgg13_bn_img_cls_torchvision",
                "pt_vgg_vgg16_img_cls_torchvision",
                "pt_vgg_vgg11_img_cls_torchvision",
                "pt_vgg_19_obj_det_hf",
                "pt_vgg_vgg13_obj_det_osmr",
                "pt_vgg_vgg16_bn_img_cls_torchvision",
                "pt_vgg_vgg11_bn_img_cls_torchvision",
                "pt_vgg_bn_vgg19b_obj_det_osmr",
                "pt_vgg_vgg16_obj_det_osmr",
                "pt_vgg_vgg19_bn_obj_det_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 784), torch.float32), ((784, 128), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128), torch.float32), ((128, 64), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 64), torch.float32), ((64, 12), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 12), torch.float32), ((12, 3), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 3), torch.float32), ((3, 12), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 12), torch.float32), ((12, 64), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 64), torch.float32), ((64, 128), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 128), torch.float32), ((128, 784), torch.float32)],
        {"model_name": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((197, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 197, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 197, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf",
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((197, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 197, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 197, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_name": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((197, 384), torch.float32), ((384, 384), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 197, 64), torch.float32), ((6, 64, 197), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((6, 197, 197), torch.float32), ((6, 197, 64), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 197, 384), torch.float32), ((384, 1536), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 197, 1536), torch.float32), ((1536, 384), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 384), torch.float32), ((384, 1000), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((197, 192), torch.float32), ((192, 192), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3, 197, 64), torch.float32), ((3, 64, 197), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3, 197, 197), torch.float32), ((3, 197, 64), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 197, 192), torch.float32), ((192, 768), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 197, 768), torch.float32), ((768, 192), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 192), torch.float32), ((192, 1000), torch.float32)],
        {"model_name": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2208), torch.float32), ((2208, 1000), torch.float32)],
        {"model_name": ["pt_densenet_densenet161_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1920), torch.float32), ((1920, 1000), torch.float32)],
        {
            "model_name": ["pt_densenet_densenet201_img_cls_torchvision", "pt_regnet_regnet_x_8gf_img_cls_torchvision"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1024), torch.float32), ((1024, 18), torch.float32)],
        {"model_name": ["pt_densenet_densenet121_hf_xray_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1664), torch.float32), ((1664, 1000), torch.float32)],
        {"model_name": ["pt_densenet_densenet169_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1280), torch.float32), ((1280, 1000), torch.float32)],
        {
            "model_name": [
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
                "pt_mobilenetv2_basic_img_cls_torchhub",
                "pt_mobilenetv2_mobilenet_v2_img_cls_torchvision",
                "pt_mobilenetv2_mobilenetv2_100_img_cls_timm",
                "pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub",
                "pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1792), torch.float32), ((1792, 1000), torch.float32)],
        {
            "model_name": [
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 19200, 64), torch.float32), ((64, 64), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((300, 64), torch.float32), ((64, 64), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 19200, 64), torch.float32), ((1, 64, 300), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 19200, 300), torch.float32), ((1, 300, 64), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 19200, 64), torch.float32), ((64, 256), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 19200, 256), torch.float32), ((256, 64), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 4800, 128), torch.float32), ((128, 128), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((300, 128), torch.float32), ((128, 128), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 4800, 64), torch.float32), ((2, 64, 300), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((2, 4800, 300), torch.float32), ((2, 300, 64), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((4800, 128), torch.float32), ((128, 128), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 4800, 128), torch.float32), ((128, 512), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 4800, 512), torch.float32), ((512, 128), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1200, 320), torch.float32), ((320, 320), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((300, 320), torch.float32), ((320, 320), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((5, 1200, 64), torch.float32), ((5, 64, 300), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((5, 1200, 300), torch.float32), ((5, 300, 64), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1200, 320), torch.float32), ((320, 320), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1200, 320), torch.float32), ((320, 1280), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1200, 1280), torch.float32), ((1280, 320), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((300, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 300, 64), torch.float32), ((8, 64, 300), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((8, 300, 300), torch.float32), ((8, 300, 64), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 300, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 300, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_name": ["pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1536), torch.float32), ((1536, 1000), torch.float32)],
        {
            "model_name": [
                "pt_inception_v4_img_cls_osmr",
                "pt_inception_inception_v4_img_cls_timm",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 768, 196), torch.float32), ((196, 384), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 768, 384), torch.float32), ((384, 196), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 196, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 196, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 11221), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024, 196), torch.float32), ((196, 512), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1024, 512), torch.float32), ((512, 196), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 196, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 196, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_name": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm", "pt_mlp_mixer_mixer_l16_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 21843), torch.float32)],
        {"model_name": ["pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 768), torch.float32), ((768, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 512), torch.float32), ((512, 256), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 256), torch.float32), ((256, 512), torch.float32)],
        {"model_name": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 512), torch.float32), ((512, 1000), torch.float32)],
        {
            "model_name": [
                "pt_mlp_mixer_base_img_cls_github",
                "pt_mobilenetv3_ssd_resnet18_img_cls_torchvision",
                "pt_mobilenetv3_ssd_resnet34_img_cls_torchvision",
                "pt_resnet_resnet18_img_cls_torchvision",
                "pt_resnet_resnet34_img_cls_torchvision",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_vovnet_vovnet27s_obj_det_osmr",
            ],
            "pcc": 0.99,
        },
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 9216), torch.float32), ((9216, 128), torch.float32)],
            {"model_name": ["pt_mnist_base_img_cls_github"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 128), torch.float32), ((128, 10), torch.float32)],
        {"model_name": ["pt_mnist_base_img_cls_github"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024), torch.float32), ((1024, 9), torch.float32)],
        {"model_name": ["pt_mobilenet_v1_basic_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 768), torch.float32), ((768, 1001), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024), torch.float32), ((1024, 1001), torch.float32)],
        {"model_name": ["pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1280), torch.float32), ((1280, 1001), torch.float32)],
        {
            "model_name": [
                "pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf",
                "pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 576), torch.float32), ((576, 1024), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 960), torch.float32), ((960, 1280), torch.float32)],
        {"model_name": ["pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 4096), torch.float32), ((4096, 2), torch.float32)],
        {"model_name": ["pt_rcnn_base_obj_det_torchvision_rect_0"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1088), torch.float32), ((1088, 1000), torch.float32)],
        {"model_name": ["pt_regnet_facebook_regnet_y_040_img_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 7392), torch.float32), ((7392, 1000), torch.float32)],
            {"model_name": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"], "pcc": 0.99},
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
    (
        Matmul1,
        [((1, 888), torch.float32), ((888, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 3712), torch.float32), ((3712, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 440), torch.float32), ((440, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2520), torch.float32), ((2520, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1008), torch.float32), ((1008, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 912), torch.float32), ((912, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_1_6gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 672), torch.float32), ((672, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 2016), torch.float32), ((2016, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 784), torch.float32), ((784, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_800mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1512), torch.float32), ((1512, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_3_2gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 400), torch.float32), ((400, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_x_400mf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 3024), torch.float32), ((3024, 1000), torch.float32)],
        {"model_name": ["pt_regnet_regnet_y_16gf_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.float32), ((64, 64), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((256, 64), torch.float32), ((64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b5_img_cls_hf",
                "pt_segformer_nvidia_mit_b4_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_mit_b2_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.float32), ((1, 64, 256), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 16384, 256), torch.float32), ((1, 256, 64), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.float32), ((64, 256), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 16384, 256), torch.float32), ((256, 64), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 4096, 128), torch.float32), ((128, 128), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((256, 128), torch.float32), ((128, 128), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((2, 4096, 64), torch.float32), ((2, 64, 256), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((2, 4096, 256), torch.float32), ((2, 256, 64), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((4096, 128), torch.float32), ((128, 128), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 4096, 128), torch.float32), ((128, 512), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 4096, 512), torch.float32), ((512, 128), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 1024, 320), torch.float32), ((320, 320), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((256, 320), torch.float32), ((320, 320), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((5, 1024, 64), torch.float32), ((5, 64, 256), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((5, 1024, 256), torch.float32), ((5, 256, 64), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1024, 320), torch.float32), ((320, 320), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 1024, 320), torch.float32), ((320, 1280), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 1024, 1280), torch.float32), ((1280, 320), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((256, 512), torch.float32), ((512, 512), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((8, 256, 64), torch.float32), ((8, 64, 256), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((8, 256, 256), torch.float32), ((8, 256, 64), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 256, 512), torch.float32), ((512, 2048), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 256, 2048), torch.float32), ((2048, 512), torch.float32)],
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
        },
    ),
    (
        Matmul1,
        [((1, 256, 512), torch.float32), ((512, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1024, 320), torch.float32), ((320, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 4096, 128), torch.float32), ((128, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 64), torch.float32), ((64, 768), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1024, 320), torch.float32), ((320, 256), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 4096, 128), torch.float32), ((128, 256), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.float32), ((32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 32), torch.float32), ((32, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.float32), ((1, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 256), torch.float32), ((1, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.float32), ((32, 128), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 16384, 128), torch.float32), ((128, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 4096, 64), torch.float32), ((64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((2, 4096, 32), torch.float32), ((2, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((2, 4096, 256), torch.float32), ((2, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((4096, 64), torch.float32), ((64, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 4096, 64), torch.float32), ((64, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 4096, 256), torch.float32), ((256, 64), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1024, 160), torch.float32), ((160, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 160), torch.float32), ((160, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((5, 1024, 32), torch.float32), ((5, 32, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((5, 1024, 256), torch.float32), ((5, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1024, 160), torch.float32), ((160, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1024, 160), torch.float32), ((160, 640), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 1024, 640), torch.float32), ((640, 160), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((256, 256), torch.float32), ((256, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((8, 256, 256), torch.float32), ((8, 256, 32), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 256), torch.float32), ((256, 1024), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256, 1024), torch.float32), ((1024, 256), torch.float32)],
        {
            "model_name": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((1, 256), torch.float32), ((256, 1000), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 256, 256), torch.float32), ((256, 256), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 1024, 160), torch.float32), ((160, 256), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 16384, 32), torch.float32), ((32, 256), torch.float32)],
        {"model_name": ["pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3136, 96), torch.float32), ((96, 288), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((192, 49, 32), torch.float32), ((192, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((192, 49, 49), torch.float32), ((192, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((3136, 96), torch.float32), ((96, 96), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((3136, 96), torch.float32), ((96, 384), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3136, 384), torch.float32), ((384, 96), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((784, 384), torch.float32), ((384, 192), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((784, 192), torch.float32), ((192, 576), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((96, 49, 32), torch.float32), ((96, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((96, 49, 49), torch.float32), ((96, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((784, 192), torch.float32), ((192, 192), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((784, 192), torch.float32), ((192, 768), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((784, 768), torch.float32), ((768, 192), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((196, 768), torch.float32), ((768, 384), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((196, 384), torch.float32), ((384, 1152), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((48, 49, 32), torch.float32), ((48, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((48, 49, 49), torch.float32), ((48, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((196, 384), torch.float32), ((384, 384), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((196, 384), torch.float32), ((384, 1536), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((196, 1536), torch.float32), ((1536, 384), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((49, 1536), torch.float32), ((1536, 768), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((49, 768), torch.float32), ((768, 2304), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((24, 49, 32), torch.float32), ((24, 32, 49), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((24, 49, 49), torch.float32), ((24, 49, 32), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((49, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_name": [
                "pt_swin_swin_s_img_cls_torchvision",
                "pt_swin_swin_t_img_cls_torchvision",
                "pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul1,
        [((49, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((49, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_swin_swin_s_img_cls_torchvision", "pt_swin_swin_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3136, 128), torch.float32), ((128, 384), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 49, 32), torch.float32), ((256, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((256, 49, 49), torch.float32), ((256, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3136, 128), torch.float32), ((128, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3136, 128), torch.float32), ((128, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((3136, 512), torch.float32), ((512, 128), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((784, 512), torch.float32), ((512, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((784, 256), torch.float32), ((256, 768), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 49, 32), torch.float32), ((128, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((128, 49, 49), torch.float32), ((128, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((784, 256), torch.float32), ((256, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((784, 256), torch.float32), ((256, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((784, 1024), torch.float32), ((1024, 256), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((196, 1024), torch.float32), ((1024, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((196, 512), torch.float32), ((512, 1536), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 49, 32), torch.float32), ((64, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((64, 49, 49), torch.float32), ((64, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((196, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((196, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((196, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((49, 2048), torch.float32), ((2048, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((49, 1024), torch.float32), ((1024, 3072), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 49, 32), torch.float32), ((32, 32, 49), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((32, 49, 49), torch.float32), ((32, 49, 32), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((49, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((49, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((49, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {"model_name": ["pt_swin_swin_b_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 3136, 96), torch.float32), ((96, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 3136, 384), torch.float32), ((384, 96), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 784, 384), torch.float32), ((384, 192), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 784, 192), torch.float32), ((192, 768), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 784, 768), torch.float32), ((768, 192), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 196, 768), torch.float32), ((768, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 196, 384), torch.float32), ((384, 1536), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 196, 1536), torch.float32), ((1536, 384), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 49, 1536), torch.float32), ((1536, 768), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 49, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul1,
        [((1, 49, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_name": ["pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf"], "pcc": 0.99},
    ),
    pytest.param(
        (
            Matmul1,
            [((1, 25088), torch.float32), ((25088, 4096), torch.float32)],
            {
                "model_name": [
                    "pt_vgg_vgg13_img_cls_torchvision",
                    "pt_vgg_vgg19_img_cls_torchvision",
                    "pt_vgg_vgg19_obj_det_osmr",
                    "pt_vgg_vgg11_obj_det_osmr",
                    "pt_vgg_bn_vgg19_obj_det_osmr",
                    "pt_vgg_vgg19_bn_obj_det_torchhub",
                    "pt_vgg_vgg13_bn_img_cls_torchvision",
                    "pt_vgg_vgg16_img_cls_torchvision",
                    "pt_vgg_vgg11_img_cls_torchvision",
                    "pt_vgg_19_obj_det_hf",
                    "pt_vgg_vgg13_obj_det_osmr",
                    "pt_vgg_vgg16_bn_img_cls_torchvision",
                    "pt_vgg_vgg11_bn_img_cls_torchvision",
                    "pt_vgg_bn_vgg19b_obj_det_osmr",
                    "pt_vgg_vgg16_obj_det_osmr",
                ],
                "pcc": 0.99,
            },
        ),
        marks=[pytest.mark.xfail(reason="Data mismatch between framework output and compiled model output")],
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes, forge_property_recorder):
    forge_property_recorder("tags.op_name", "Matmul")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")

    for metadata_name, metadata_value in metadata.items():
        forge_property_recorder("tags." + str(metadata_name), metadata_value)

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

    compiled_model = compile(framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder)

    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
        forge_property_handler=forge_property_recorder,
    )
