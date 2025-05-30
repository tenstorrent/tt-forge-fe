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


class Embedding0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding0.weight_1",
            forge.Parameter(*(32000, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding0.weight_1"))
        return embedding_output_1


class Embedding1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding1.weight_1",
            forge.Parameter(*(2, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding1.weight_1"))
        return embedding_output_1


class Embedding2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, embedding_input_0, embedding_input_1):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, embedding_input_1)
        return embedding_output_1


class Embedding3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding3.weight_1",
            forge.Parameter(*(30522, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding3.weight_1"))
        return embedding_output_1


class Embedding4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding4.weight_1",
            forge.Parameter(*(512, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding4.weight_1"))
        return embedding_output_1


class Embedding5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding5.weight_1",
            forge.Parameter(*(21128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding5.weight_1"))
        return embedding_output_1


class Embedding6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding6_const_1", shape=(18000, 768), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding6_const_1"))
        return embedding_output_1


class Embedding7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding7.weight_1",
            forge.Parameter(*(513, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding7.weight_1"))
        return embedding_output_1


class Embedding8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding8.weight_1",
            forge.Parameter(*(30000, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding8.weight_1"))
        return embedding_output_1


class Embedding9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding9.weight_1",
            forge.Parameter(*(2, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding9.weight_1"))
        return embedding_output_1


class Embedding10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding10.weight_1",
            forge.Parameter(*(512, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding10.weight_1"))
        return embedding_output_1


class Embedding11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding11.weight_1",
            forge.Parameter(*(50265, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding11.weight_1"))
        return embedding_output_1


class Embedding12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding12.weight_1",
            forge.Parameter(*(28996, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding12.weight_1"))
        return embedding_output_1


class Embedding13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding13.weight_1",
            forge.Parameter(*(2, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding13.weight_1"))
        return embedding_output_1


class Embedding14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding14.weight_1",
            forge.Parameter(*(512, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding14.weight_1"))
        return embedding_output_1


class Embedding15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding15.weight_1",
            forge.Parameter(*(51200, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding15.weight_1"))
        return embedding_output_1


class Embedding16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding16.weight_1",
            forge.Parameter(*(50257, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding16.weight_1"))
        return embedding_output_1


class Embedding17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding17.weight_1",
            forge.Parameter(*(128256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding17.weight_1"))
        return embedding_output_1


class Embedding18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding18.weight_1",
            forge.Parameter(*(50272, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding18.weight_1"))
        return embedding_output_1


class Embedding19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding19.weight_1",
            forge.Parameter(*(2050, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding19.weight_1"))
        return embedding_output_1


class Embedding20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding20.weight_1",
            forge.Parameter(*(51200, 2560), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding20.weight_1"))
        return embedding_output_1


class Embedding21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding21.weight_1",
            forge.Parameter(*(51865, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding21.weight_1"))
        return embedding_output_1


class Embedding22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding22.weight_1",
            forge.Parameter(*(30522, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding22.weight_1"))
        return embedding_output_1


class Embedding23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding23.weight_1",
            forge.Parameter(*(30522, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding23.weight_1"))
        return embedding_output_1


class Embedding24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding24_const_1", shape=(21128, 768), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding24_const_1"))
        return embedding_output_1


class Embedding25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding25.weight_1",
            forge.Parameter(*(250880, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding25.weight_1"))
        return embedding_output_1


class Embedding26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding26.weight_1",
            forge.Parameter(*(49408, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding26.weight_1"))
        return embedding_output_1


class Embedding27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding27.weight_1",
            forge.Parameter(*(77, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding27.weight_1"))
        return embedding_output_1


class Embedding28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding28.weight_1",
            forge.Parameter(*(28996, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding28.weight_1"))
        return embedding_output_1


class Embedding29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding29.weight_1",
            forge.Parameter(*(50257, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding29.weight_1"))
        return embedding_output_1


class Embedding30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding30.weight_1",
            forge.Parameter(*(50272, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding30.weight_1"))
        return embedding_output_1


class Embedding31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding31.weight_1",
            forge.Parameter(*(2050, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding31.weight_1"))
        return embedding_output_1


class Embedding32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding32.weight_1",
            forge.Parameter(*(50272, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding32.weight_1"))
        return embedding_output_1


class Embedding33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding33.weight_1",
            forge.Parameter(*(2050, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding33.weight_1"))
        return embedding_output_1


class Embedding34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding34.weight_1",
            forge.Parameter(*(51200, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding34.weight_1"))
        return embedding_output_1


class Embedding35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding35.weight_1",
            forge.Parameter(*(51865, 384), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding35.weight_1"))
        return embedding_output_1


class Embedding36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("embedding36_const_1", shape=(21128, 128), dtype=torch.float32)

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_constant("embedding36_const_1"))
        return embedding_output_1


class Embedding37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding37.weight_1",
            forge.Parameter(*(51866, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding37.weight_1"))
        return embedding_output_1


class Embedding38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding38.weight_1",
            forge.Parameter(*(32256, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding38.weight_1"))
        return embedding_output_1


class Embedding39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding39.weight_1",
            forge.Parameter(*(119547, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding39.weight_1"))
        return embedding_output_1


class Embedding40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding40.weight_1",
            forge.Parameter(*(50280, 1024), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding40.weight_1"))
        return embedding_output_1


class Embedding41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding41.weight_1",
            forge.Parameter(*(2049, 1536), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding41.weight_1"))
        return embedding_output_1


class Embedding42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding42.weight_1",
            forge.Parameter(*(32128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding42.weight_1"))
        return embedding_output_1


class Embedding43(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "embedding43.weight_1",
            forge.Parameter(*(51865, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, embedding_input_0):
        embedding_output_1 = forge.op.Embedding("", embedding_input_0, self.get_parameter("embedding43.weight_1"))
        return embedding_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Embedding0,
        [((1, 6), torch.int64)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 31999,
        },
    ),
    (
        Embedding1,
        [((1, 6), torch.int64)],
        {
            "model_names": [
                "onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
                "pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding2,
        [((1, 6), torch.int64), ((512, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding3,
        [((1, 9), torch.int64)],
        {
            "model_names": ["pd_bert_bert_base_uncased_mlm_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding4,
        [((1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding1,
        [((1, 9), torch.int64)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_mlm_padlenlp",
                "pd_ernie_1_0_mlm_padlenlp",
                "pd_ernie_1_0_qa_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_qa_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding5,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 21127,
        },
    ),
    (
        Embedding4,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_roberta_rbt4_ch_clm_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding1,
        [((1, 11), torch.int64)],
        {
            "model_names": [
                "pd_bert_chinese_roberta_base_qa_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding6,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_ernie_1_0_mlm_padlenlp", "pd_ernie_1_0_qa_padlenlp"], "pcc": 0.99, "max_int": 17999},
    ),
    (
        Embedding7,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_ernie_1_0_mlm_padlenlp", "pd_ernie_1_0_qa_padlenlp"], "pcc": 0.99, "max_int": 512},
    ),
    (
        Embedding8,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 29999,
        },
    ),
    (
        Embedding9,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding10,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding11,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 50264},
    ),
    (
        Embedding2,
        [((1, 256), torch.int64), ((1026, 1024), torch.float32)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99, "max_int": 1025},
    ),
    (
        Embedding12,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 28995,
        },
    ),
    (
        Embedding13,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding14,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding4,
        [((1, 6), torch.int64)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding15,
        [((1, 256), torch.int64)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "max_int": 51199,
        },
    ),
    (
        Embedding3,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "onnx_bert_bert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding4,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding1,
        [((1, 128), torch.int64)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding16,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding2,
        [((1, 32), torch.int64), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding17,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99, "max_int": 128255},
    ),
    (
        Embedding18,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding19,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding20,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "max_int": 51199},
    ),
    (
        Embedding21,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding22,
        [((1, 384), torch.int64)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding2,
        [((1, 384), torch.int64), ((2, 1024), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding2,
        [((1, 384), torch.int64), ((512, 1024), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding23,
        [((1, 13), torch.int64)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "max_int": 30521,
        },
    ),
    (
        Embedding2,
        [((1, 13), torch.int64), ((2, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding2,
        [((1, 13), torch.int64), ((512, 384), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding0,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding4,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding1,
        [((1, 10), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_mlm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding3,
        [((1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 30521},
    ),
    (
        Embedding4,
        [((1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding1,
        [((1, 8), torch.int64)],
        {"model_names": ["pd_bert_bert_base_uncased_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding24,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding2,
        [((1, 11), torch.int64), ((2, 768), torch.float32)],
        {"model_names": ["pd_roberta_rbt4_ch_clm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding25,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"], "pcc": 0.99, "max_int": 250879},
    ),
    (
        Embedding26,
        [((2, 7), torch.int64)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "max_int": 49407},
    ),
    (
        Embedding27,
        [((1, 7), torch.int64)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "max_int": 76},
    ),
    (
        Embedding28,
        [((1, 384), torch.int64)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99, "max_int": 28995},
    ),
    (
        Embedding4,
        [((1, 384), torch.int64)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding29,
        [((1, 7), torch.int64)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 50256,
        },
    ),
    (
        Embedding2,
        [((1, 7), torch.int64), ((1024, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "max_int": 1023,
        },
    ),
    (
        Embedding17,
        [((1, 4), torch.int64)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99, "max_int": 128255},
    ),
    (
        Embedding30,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding31,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding32,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "max_int": 50271},
    ),
    (
        Embedding33,
        [((1, 32), torch.int64)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "max_int": 2049},
    ),
    (
        Embedding34,
        [((1, 7), torch.int64)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "max_int": 51199},
    ),
    (
        Embedding35,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
    (
        Embedding36,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding9,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding10,
        [((1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding0,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 31999},
    ),
    (
        Embedding4,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding1,
        [((1, 15), torch.int64)],
        {"model_names": ["pd_bert_bert_base_japanese_seq_cls_padlenlp"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding5,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"], "pcc": 0.99, "max_int": 21127},
    ),
    (
        Embedding37,
        [((1, 2), torch.int64)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf", "onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "max_int": 51865,
        },
    ),
    (
        Embedding2,
        [((1, 128), torch.int64), ((2, 768), torch.float32)],
        {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding2,
        [((1, 128), torch.int64), ((512, 768), torch.float32)],
        {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding2,
        [((1, 2), torch.int64), ((448, 1280), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "max_int": 447},
    ),
    (
        Embedding5,
        [((1, 8), torch.int64)],
        {
            "model_names": ["pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 21127,
        },
    ),
    (
        Embedding2,
        [((1, 8), torch.int64), ((512, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 511,
        },
    ),
    (
        Embedding2,
        [((1, 8), torch.int64), ((2, 768), torch.float32)],
        {
            "model_names": ["pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp"],
            "pcc": 0.99,
            "max_int": 1,
        },
    ),
    (
        Embedding8,
        [((1, 14), torch.int64)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "max_int": 29999},
    ),
    (
        Embedding9,
        [((1, 14), torch.int64)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding10,
        [((1, 14), torch.int64)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding13,
        [((1, 384), torch.int64)],
        {"model_names": ["pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99, "max_int": 1},
    ),
    (
        Embedding14,
        [((1, 384), torch.int64)],
        {"model_names": ["pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99, "max_int": 511},
    ),
    (
        Embedding38,
        [((1, 588), torch.int64)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99, "max_int": 32255},
    ),
    (
        Embedding39,
        [((1, 128), torch.int64)],
        {
            "model_names": ["pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf"],
            "pcc": 0.99,
            "max_int": 119546,
        },
    ),
    (
        Embedding28,
        [((1, 128), torch.int64)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99, "max_int": 28995},
    ),
    (
        Embedding16,
        [((1, 256), torch.int64)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "max_int": 50256},
    ),
    (
        Embedding2,
        [((1, 256), torch.int64), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"], "pcc": 0.99, "max_int": 2047},
    ),
    (
        Embedding40,
        [((1, 6), torch.int64)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "max_int": 50279},
    ),
    (
        Embedding41,
        [((2, 1), torch.int64)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "max_int": 2048},
    ),
    (
        Embedding42,
        [((2, 13), torch.int64)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "max_int": 32127},
    ),
    (
        Embedding2,
        [((13, 13), torch.int32), ((32, 12), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "max_int": 31},
    ),
    (
        Embedding43,
        [((1, 1), torch.int64)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "max_int": 51864},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Embedding")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.pop("pcc")
    max_int = metadata.pop("max_int")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

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
