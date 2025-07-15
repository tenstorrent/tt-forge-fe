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


class Matmul0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul0.weight_1",
            forge.Parameter(*(128, 312), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul0.weight_1"))
        return matmul_output_1


class Matmul1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul1.weight_1",
            forge.Parameter(*(312, 312), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul1.weight_1"))
        return matmul_output_1


class Matmul2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, matmul_input_0, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, matmul_input_1)
        return matmul_output_1


class Matmul3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul3.weight_1",
            forge.Parameter(*(312, 1248), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul3.weight_1"))
        return matmul_output_1


class Matmul4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul4.weight_1",
            forge.Parameter(*(1248, 312), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul4.weight_1"))
        return matmul_output_1


class Matmul5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul5.weight_1",
            forge.Parameter(*(312, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul5.weight_1"))
        return matmul_output_1


class Matmul6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul6.weight_1",
            forge.Parameter(*(2048, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul6.weight_1"))
        return matmul_output_1


class Matmul7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul7_const_1", shape=(1, 1, 4), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul7_const_1"))
        return matmul_output_1


class Matmul8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul8_const_1", shape=(1, 1, 256), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul8_const_1"))
        return matmul_output_1


class Matmul9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul9_const_1", shape=(1, 1, 5), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul9_const_1"))
        return matmul_output_1


class Matmul10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul10.weight_1",
            forge.Parameter(*(64, 64), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul10.weight_1"))
        return matmul_output_1


class Matmul11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul11.weight_1",
            forge.Parameter(*(64, 256), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul11.weight_1"))
        return matmul_output_1


class Matmul12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul12.weight_1",
            forge.Parameter(*(256, 64), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul12.weight_1"))
        return matmul_output_1


class Matmul13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul13.weight_1",
            forge.Parameter(*(128, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul13.weight_1"))
        return matmul_output_1


class Matmul14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul14.weight_1",
            forge.Parameter(*(128, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul14.weight_1"))
        return matmul_output_1


class Matmul15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul15.weight_1",
            forge.Parameter(*(512, 128), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul15.weight_1"))
        return matmul_output_1


class Matmul16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul16.weight_1",
            forge.Parameter(*(320, 320), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul16.weight_1"))
        return matmul_output_1


class Matmul17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul17.weight_1",
            forge.Parameter(*(320, 1280), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul17.weight_1"))
        return matmul_output_1


class Matmul18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul18.weight_1",
            forge.Parameter(*(1280, 320), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul18.weight_1"))
        return matmul_output_1


class Matmul19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul19.weight_1",
            forge.Parameter(*(512, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul19.weight_1"))
        return matmul_output_1


class Matmul20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul20.weight_1",
            forge.Parameter(*(512, 2048), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul20.weight_1"))
        return matmul_output_1


class Matmul21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul21.weight_1",
            forge.Parameter(*(2048, 512), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul21.weight_1"))
        return matmul_output_1


class Matmul22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul22.weight_1",
            forge.Parameter(*(512, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul22.weight_1"))
        return matmul_output_1


class Matmul23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul23.weight_1",
            forge.Parameter(*(320, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul23.weight_1"))
        return matmul_output_1


class Matmul24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul24.weight_1",
            forge.Parameter(*(128, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul24.weight_1"))
        return matmul_output_1


class Matmul25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul25.weight_1",
            forge.Parameter(*(64, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul25.weight_1"))
        return matmul_output_1


class Matmul26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul26.weight_1",
            forge.Parameter(*(768, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul26.weight_1"))
        return matmul_output_1


class Matmul27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul27.weight_1",
            forge.Parameter(*(768, 3072), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul27.weight_1"))
        return matmul_output_1


class Matmul28(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul28.weight_1",
            forge.Parameter(*(3072, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul28.weight_1"))
        return matmul_output_1


class Matmul29(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul29.weight_1",
            forge.Parameter(*(768, 2), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul29.weight_1"))
        return matmul_output_1


class Matmul30(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul30_const_1", shape=(1, 1, 11), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul30_const_1"))
        return matmul_output_1


class Matmul31(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul31_const_1", shape=(1, 1, 6), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul31_const_1"))
        return matmul_output_1


class Matmul32(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul32_const_1", shape=(1, 1, 35), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul32_const_1"))
        return matmul_output_1


class Matmul33(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul33_const_1", shape=(1, 1, 29), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul33_const_1"))
        return matmul_output_1


class Matmul34(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul34.weight_1",
            forge.Parameter(*(192, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul34.weight_1"))
        return matmul_output_1


class Matmul35(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul35.weight_1",
            forge.Parameter(*(192, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul35.weight_1"))
        return matmul_output_1


class Matmul36(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul36.weight_1",
            forge.Parameter(*(768, 192), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul36.weight_1"))
        return matmul_output_1


class Matmul37(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul37.weight_1",
            forge.Parameter(*(9216, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul37.weight_1"))
        return matmul_output_1


class Matmul38(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul38.weight_1",
            forge.Parameter(*(4096, 4096), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul38.weight_1"))
        return matmul_output_1


class Matmul39(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul39.weight_1",
            forge.Parameter(*(4096, 1000), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul39.weight_1"))
        return matmul_output_1


class Matmul40(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul40_const_0", shape=(1, 48), dtype=torch.float32)

    def forward(self, matmul_input_1):
        matmul_output_1 = forge.op.Matmul("", self.get_constant("matmul40_const_0"), matmul_input_1)
        return matmul_output_1


class Matmul41(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "matmul41.weight_1",
            forge.Parameter(*(96, 6625), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_parameter("matmul41.weight_1"))
        return matmul_output_1


class Matmul42(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("matmul42_const_1", shape=(1, 1, 12), dtype=torch.float32)

    def forward(self, matmul_input_0):
        matmul_output_1 = forge.op.Matmul("", matmul_input_0, self.get_constant("matmul42_const_1"))
        return matmul_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Matmul0, [((1, 11, 128), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Matmul1, [((11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Matmul2,
        [((12, 11, 26), torch.float32), ((12, 26, 11), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 11, 11), torch.float32), ((12, 11, 26), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (Matmul3, [((1, 11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Matmul4, [((1, 11, 1248), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (Matmul5, [((1, 11, 312), torch.float32)], {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99}),
    (
        Matmul2,
        [((1, 11, 128), torch.float32), ((128, 21128), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul6,
        [((1, 2048), torch.float32)],
        {
            "model_names": [
                "pd_resnet_101_img_cls_paddlemodels",
                "pd_resnet_152_img_cls_paddlemodels",
                "tf_resnet_resnet50_img_cls_keras",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 128, 128), torch.float32), ((128, 1024), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((128, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((16, 128, 64), torch.float32), ((16, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((16, 128, 128), torch.float32), ((16, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 128, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 128, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 128, 1024), torch.float32), ((1024, 2), torch.float32)],
        {"model_names": ["pt_albert_large_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((128, 768), torch.float32), ((768, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((12, 128, 64), torch.float32), ((12, 64, 128), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((12, 128, 128), torch.float32), ((12, 128, 64), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 128, 768), torch.float32), ((768, 3072), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 128, 3072), torch.float32), ((3072, 768), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((128, 768), torch.float32), ((768, 1), torch.float32)],
        {"model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 768), torch.float32), ((768, 1), torch.float32)],
        {"model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 1536), torch.bfloat16), ((1536, 1000), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b3_img_cls_torchvision",
                "pt_inception_inception_v4_tf_in1k_img_cls_timm",
                "pt_inception_inception_v4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1792), torch.bfloat16), ((1792, 1000), torch.bfloat16)],
        {
            "model_names": [
                "pt_efficientnet_efficientnet_b4_img_cls_torchvision",
                "pt_efficientnet_efficientnet_b4_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 2304), torch.bfloat16), ((2304, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b6_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 2048), torch.bfloat16), ((2048, 1000), torch.bfloat16)],
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
        },
    ),
    (
        Matmul2,
        [((4, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul7,
        [((1, 32, 1), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((4, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 4, 64), torch.float32), ((32, 64, 4), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 4, 4), torch.float32), ((32, 4, 64), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((4, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 4, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 4, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 768), torch.bfloat16), ((768, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 256, 512), torch.bfloat16), ((512, 256), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 256, 256), torch.bfloat16), ((256, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_base_img_cls_github"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 512), torch.bfloat16), ((512, 1000), torch.bfloat16)],
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
        },
    ),
    (
        Matmul2,
        [((1, 512, 196), torch.bfloat16), ((196, 256), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 512, 256), torch.bfloat16), ((256, 196), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 196, 512), torch.bfloat16), ((512, 2048), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 196, 2048), torch.bfloat16), ((2048, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 1024), torch.bfloat16), ((1024, 1001), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_1_0_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024), torch.bfloat16), ((1024, 1000), torch.bfloat16)],
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
        },
    ),
    (
        Matmul2,
        [((256, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (Matmul8, [((1, 16, 1), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99}),
    (
        Matmul2,
        [((32, 256, 64), torch.float32), ((32, 64, 256), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 256, 256), torch.float32), ((32, 256, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 256, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 256, 2048), torch.float32), ((2048, 51200), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((5, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (Matmul9, [((1, 16, 1), torch.float32)], {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99}),
    (
        Matmul2,
        [((32, 5, 64), torch.float32), ((32, 64, 5), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 5, 5), torch.float32), ((32, 5, 64), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((5, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 5, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 5, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 7392), torch.bfloat16), ((7392, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_128gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 2016), torch.bfloat16), ((2016, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_y_8gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 16384, 32), torch.bfloat16), ((32, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((256, 32), torch.bfloat16), ((32, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 16384, 32), torch.bfloat16), ((1, 32, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 16384, 256), torch.bfloat16), ((1, 256, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 16384, 32), torch.bfloat16), ((32, 128), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 16384, 128), torch.bfloat16), ((128, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 4096, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((256, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((2, 4096, 32), torch.bfloat16), ((2, 32, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((2, 4096, 256), torch.bfloat16), ((2, 256, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((4096, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 4096, 64), torch.bfloat16), ((64, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 4096, 256), torch.bfloat16), ((256, 64), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 1024, 160), torch.bfloat16), ((160, 160), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((256, 160), torch.bfloat16), ((160, 160), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((5, 1024, 32), torch.bfloat16), ((5, 32, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((5, 1024, 256), torch.bfloat16), ((5, 256, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1024, 160), torch.bfloat16), ((160, 160), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 1024, 160), torch.bfloat16), ((160, 640), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 1024, 640), torch.bfloat16), ((640, 160), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((256, 256), torch.bfloat16), ((256, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((8, 256, 32), torch.bfloat16), ((8, 32, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((8, 256, 256), torch.bfloat16), ((8, 256, 32), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 256, 256), torch.bfloat16), ((256, 1024), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 256, 1024), torch.bfloat16), ((1024, 256), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 256), torch.bfloat16), ((256, 1000), torch.bfloat16)],
        {"model_names": ["pt_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((3136, 96), torch.bfloat16), ((96, 288), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((192, 49, 32), torch.bfloat16), ((192, 32, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((192, 49, 49), torch.bfloat16), ((192, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((3136, 96), torch.bfloat16), ((96, 96), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((3136, 96), torch.bfloat16), ((96, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((3136, 384), torch.bfloat16), ((384, 96), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((784, 384), torch.bfloat16), ((384, 192), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((784, 192), torch.bfloat16), ((192, 576), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((96, 49, 32), torch.bfloat16), ((96, 32, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((96, 49, 49), torch.bfloat16), ((96, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((784, 192), torch.bfloat16), ((192, 192), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((784, 192), torch.bfloat16), ((192, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((784, 768), torch.bfloat16), ((768, 192), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((196, 768), torch.bfloat16), ((768, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((196, 384), torch.bfloat16), ((384, 1152), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((48, 49, 32), torch.bfloat16), ((48, 32, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((48, 49, 49), torch.bfloat16), ((48, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((196, 384), torch.bfloat16), ((384, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((196, 384), torch.bfloat16), ((384, 1536), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((196, 1536), torch.bfloat16), ((1536, 384), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((49, 1536), torch.bfloat16), ((1536, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((49, 768), torch.bfloat16), ((768, 2304), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((24, 49, 32), torch.bfloat16), ((24, 32, 49), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((24, 49, 49), torch.bfloat16), ((24, 49, 32), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((49, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((49, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((49, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
        {"model_names": ["pt_swin_swin_s_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 768), torch.bfloat16), ((768, 1000), torch.bfloat16)],
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
        },
    ),
    (
        Matmul2,
        [((1, 25088), torch.bfloat16), ((25088, 4096), torch.bfloat16)],
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
        },
    ),
    (
        Matmul2,
        [((1, 4096), torch.bfloat16), ((4096, 4096), torch.bfloat16)],
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
        },
    ),
    (
        Matmul2,
        [((1, 4096), torch.bfloat16), ((4096, 1000), torch.bfloat16)],
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
        },
    ),
    (
        Matmul2,
        [((197, 768), torch.bfloat16), ((768, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((12, 197, 64), torch.bfloat16), ((12, 64, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((12, 197, 197), torch.bfloat16), ((12, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 197, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 197, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((50, 1024), torch.bfloat16), ((1024, 3072), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((16, 50, 64), torch.bfloat16), ((16, 64, 50), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((16, 50, 50), torch.bfloat16), ((16, 50, 64), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((50, 1024), torch.bfloat16), ((1024, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 50, 1024), torch.bfloat16), ((1024, 4096), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 50, 4096), torch.bfloat16), ((4096, 1024), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_32_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul10,
        [((1, 16384, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul10,
        [((256, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 16384, 64), torch.float32), ((1, 64, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 16384, 256), torch.float32), ((1, 256, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul11,
        [((1, 16384, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul12,
        [((1, 16384, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul13,
        [((1, 4096, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul13,
        [((256, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((2, 4096, 64), torch.float32), ((2, 64, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((2, 4096, 256), torch.float32), ((2, 256, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul13,
        [((4096, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul14,
        [((1, 4096, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul15,
        [((1, 4096, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul16,
        [((1, 1024, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul16,
        [((256, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((5, 1024, 64), torch.float32), ((5, 64, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((5, 1024, 256), torch.float32), ((5, 256, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul16,
        [((1024, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul17,
        [((1, 1024, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul18,
        [((1, 1024, 1280), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul19,
        [((256, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((8, 256, 64), torch.float32), ((8, 64, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((8, 256, 256), torch.float32), ((8, 256, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul20,
        [((1, 256, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul21,
        [((1, 256, 2048), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul22,
        [((1, 256, 512), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul23,
        [((1, 1024, 320), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul24,
        [((1, 4096, 128), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul25,
        [((1, 16384, 64), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"], "pcc": 0.99},
    ),
    (
        Matmul26,
        [((9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((12, 9, 64), torch.float32), ((12, 64, 9), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((12, 9, 9), torch.float32), ((12, 9, 64), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul27,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul28,
        [((1, 9, 3072), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (Matmul26, [((1, 768), torch.float32)], {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99}),
    (Matmul29, [((1, 768), torch.float32)], {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99}),
    (
        Matmul2,
        [((197, 1024), torch.bfloat16), ((1024, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((16, 197, 64), torch.bfloat16), ((16, 64, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((16, 197, 197), torch.bfloat16), ((16, 197, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 197, 1024), torch.bfloat16), ((1024, 4096), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 197, 4096), torch.bfloat16), ((4096, 1024), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024), torch.bfloat16), ((1024, 18), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet121_hf_xray_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 2560), torch.bfloat16), ((2560, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b7_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1280), torch.bfloat16), ((1280, 1000), torch.bfloat16)],
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
        },
    ),
    (
        Matmul2,
        [((1, 768, 196), torch.bfloat16), ((196, 384), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 768, 384), torch.bfloat16), ((384, 196), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 196, 768), torch.bfloat16), ((768, 3072), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 196, 3072), torch.bfloat16), ((3072, 768), torch.bfloat16)],
        {
            "model_names": [
                "pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 512, 49), torch.bfloat16), ((49, 256), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 512, 256), torch.bfloat16), ((256, 49), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 49, 512), torch.bfloat16), ((512, 2048), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 49, 2048), torch.bfloat16), ((2048, 512), torch.bfloat16)],
        {"model_names": ["pt_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((1, 768), torch.bfloat16), ((768, 1001), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv1_google_mobilenet_v1_0_75_192_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1280), torch.bfloat16), ((1280, 1001), torch.bfloat16)],
        {
            "model_names": ["pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((11, 2560), torch.float32), ((2560, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul30,
        [((1, 16, 1), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 11, 80), torch.float32), ((32, 80, 11), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 11, 11), torch.float32), ((32, 11, 80), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((11, 2560), torch.float32), ((2560, 10240), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 11, 10240), torch.float32), ((10240, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 11, 2560), torch.float32), ((2560, 2), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((6, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Matmul31, [((1, 32, 1), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Matmul2,
        [((16, 6, 64), torch.float32), ((16, 64, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((16, 6, 6), torch.float32), ((16, 6, 64), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((6, 1024), torch.float32), ((1024, 2816), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 6, 2816), torch.float32), ((2816, 1024), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 6, 1024), torch.float32), ((1024, 151936), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((35, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul32,
        [((1, 64, 1), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((35, 1536), torch.float32), ((1536, 256), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 35, 128), torch.float32), ((12, 128, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 35, 35), torch.float32), ((12, 35, 128), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((35, 1536), torch.float32), ((1536, 8960), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 35, 8960), torch.float32), ((8960, 1536), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 35, 1536), torch.float32), ((1536, 151936), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((29, 1536), torch.float32), ((1536, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (Matmul33, [((1, 64, 1), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99}),
    (
        Matmul2,
        [((29, 1536), torch.float32), ((1536, 256), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 29, 128), torch.float32), ((12, 128, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 29, 29), torch.float32), ((12, 29, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((29, 1536), torch.float32), ((1536, 8960), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 29, 8960), torch.float32), ((8960, 1536), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 29, 1536), torch.float32), ((1536, 151936), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 3024), torch.bfloat16), ((3024, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_160_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 3712), torch.bfloat16), ((3712, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_facebook_regnet_y_320_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 2520), torch.bfloat16), ((2520, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_32gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1008), torch.bfloat16), ((1008, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_regnet_regnet_x_3_2gf_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((4096, 96), torch.float32), ((96, 288), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((192, 64, 32), torch.float32), ((192, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((225, 2), torch.float32), ((2, 512), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((225, 512), torch.float32), ((512, 3), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((192, 64, 64), torch.float32), ((192, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((4096, 96), torch.float32), ((96, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((4096, 96), torch.float32), ((96, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((4096, 384), torch.float32), ((384, 96), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1024, 384), torch.float32), ((384, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1024, 192), torch.float32), ((192, 576), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((96, 64, 32), torch.float32), ((96, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((225, 512), torch.float32), ((512, 6), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((96, 64, 64), torch.float32), ((96, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1024, 192), torch.float32), ((192, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1024, 192), torch.float32), ((192, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1024, 768), torch.float32), ((768, 192), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 768), torch.float32), ((768, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 384), torch.float32), ((384, 1152), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((48, 64, 32), torch.float32), ((48, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((225, 512), torch.float32), ((512, 12), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((48, 64, 64), torch.float32), ((48, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 384), torch.float32), ((384, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 384), torch.float32), ((384, 1536), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 1536), torch.float32), ((1536, 384), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((64, 1536), torch.float32), ((1536, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((64, 768), torch.float32), ((768, 2304), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((24, 64, 32), torch.float32), ((24, 32, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((225, 512), torch.float32), ((512, 24), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((24, 64, 64), torch.float32), ((24, 64, 32), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((64, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((64, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((64, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 768), torch.float32), ((768, 1000), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((513, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 513, 64), torch.float32), ((12, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 513, 513), torch.float32), ((12, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((61, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 61, 64), torch.float32), ((12, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 61, 61), torch.float32), ((12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 61, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 61, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 513, 64), torch.float32), ((12, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 513, 61), torch.float32), ((12, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 513, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 513, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 513, 768), torch.float32), ((768, 32128), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 2048), torch.float32), ((2048, 1000), torch.float32)],
        {"model_names": ["onnx_xception_xception71_tf_in1k_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128, 128), torch.float32), ((128, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((128, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((16, 128, 128), torch.float32), ((16, 128, 128), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_names": ["pt_albert_xlarge_v2_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((197, 192), torch.bfloat16), ((192, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((3, 197, 64), torch.bfloat16), ((3, 64, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((3, 197, 197), torch.bfloat16), ((3, 197, 64), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 197, 192), torch.bfloat16), ((192, 768), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 197, 768), torch.bfloat16), ((768, 192), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 192), torch.bfloat16), ((192, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024, 196), torch.bfloat16), ((196, 512), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024, 512), torch.bfloat16), ((512, 196), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 196, 1024), torch.bfloat16), ((1024, 4096), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 196, 4096), torch.bfloat16), ((4096, 1024), torch.bfloat16)],
        {
            "model_names": ["pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024), torch.bfloat16), ((1024, 9), torch.bfloat16)],
        {"model_names": ["pt_mobilenetv1_basic_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((29, 896), torch.float32), ((896, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (Matmul33, [((1, 32, 1), torch.float32)], {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Matmul2,
        [((29, 896), torch.float32), ((896, 128), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((14, 29, 64), torch.float32), ((14, 64, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((14, 29, 29), torch.float32), ((14, 29, 64), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((29, 896), torch.float32), ((896, 4864), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 29, 4864), torch.float32), ((4864, 896), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 29, 896), torch.float32), ((896, 151936), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 16384, 64), torch.bfloat16), ((64, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 16384, 64), torch.bfloat16), ((1, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 16384, 256), torch.bfloat16), ((1, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 16384, 64), torch.bfloat16), ((64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 16384, 256), torch.bfloat16), ((256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 4096, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((256, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((2, 4096, 64), torch.bfloat16), ((2, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((2, 4096, 256), torch.bfloat16), ((2, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((4096, 128), torch.bfloat16), ((128, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 4096, 128), torch.bfloat16), ((128, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 4096, 512), torch.bfloat16), ((512, 128), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((256, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((5, 1024, 64), torch.bfloat16), ((5, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((5, 1024, 256), torch.bfloat16), ((5, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1024, 320), torch.bfloat16), ((320, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024, 320), torch.bfloat16), ((320, 1280), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024, 1280), torch.bfloat16), ((1280, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((256, 512), torch.bfloat16), ((512, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((8, 256, 64), torch.bfloat16), ((8, 64, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((8, 256, 256), torch.bfloat16), ((8, 256, 64), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 256, 512), torch.bfloat16), ((512, 2048), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 256, 2048), torch.bfloat16), ((2048, 512), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 256, 512), torch.bfloat16), ((512, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1024, 320), torch.bfloat16), ((320, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 4096, 128), torch.bfloat16), ((128, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 16384, 64), torch.bfloat16), ((64, 768), torch.bfloat16)],
        {
            "model_names": ["pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 768), torch.float32), ((768, 3), torch.float32)],
        {"model_names": ["pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((197, 768), torch.bfloat16), ((768, 2304), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_b_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((197, 1024), torch.bfloat16), ((1024, 3072), torch.bfloat16)],
        {"model_names": ["pt_vit_vit_l_16_img_cls_torchvision"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Matmul2,
        [((256, 1024), torch.float32), ((1024, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((16, 256, 64), torch.float32), ((16, 64, 256), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((16, 256, 256), torch.float32), ((16, 256, 64), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 256, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 256, 4096), torch.float32), ((4096, 1024), torch.float32)],
        {
            "model_names": ["pt_xglm_facebook_xglm_564m_clm_hf", "pt_codegen_salesforce_codegen_350m_nl_clm_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 256, 1024), torch.float32), ((1024, 256008), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul26,
        [((1, 9, 768), torch.float32)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp", "pd_bert_bert_base_uncased_mlm_padlenlp"],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 9, 768), torch.float32), ((768, 21128), torch.float32)],
        {"model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128, 768), torch.float32), ((768, 28996), torch.float32)],
        {"model_names": ["pt_distilbert_distilbert_base_cased_mlm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 576), torch.float32), ((576, 1024), torch.float32)],
        {"model_names": ["onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 1024), torch.float32), ((1024, 1000), torch.float32)],
        {
            "model_names": [
                "onnx_mobilenetv3_mobilenet_v3_small_img_cls_torchhub",
                "onnx_vovnet_vovnet_v1_57_obj_det_torchhub",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Matmul2,
        [((1, 512), torch.float32), ((512, 1000), torch.float32)],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 1280), torch.float32), ((1280, 1000), torch.float32)],
        {"model_names": ["onnx_mobilenetv2_mobilenetv2_110d_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 9, 768), torch.float32), ((768, 30522), torch.float32)],
        {"model_names": ["pd_bert_bert_base_uncased_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Matmul34,
        [((197, 192), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((3, 197, 64), torch.float32), ((3, 64, 197), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((3, 197, 197), torch.float32), ((3, 197, 64), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul35,
        [((1, 197, 192), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul36,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 192), torch.float32), ((192, 1000), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (Matmul37, [((1, 9216), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (Matmul38, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (Matmul39, [((1, 4096), torch.float32)], {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99}),
    (
        Matmul2,
        [((1, 288), torch.float32), ((288, 192), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul40,
        [((48, 192), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 48), torch.float32), ((48, 192), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 96), torch.float32), ((96, 192), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul41,
        [((1, 25, 96), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 784), torch.float32), ((784, 128), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128), torch.float32), ((128, 64), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 64), torch.float32), ((64, 12), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 12), torch.float32), ((12, 3), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 3), torch.float32), ((3, 12), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 12), torch.float32), ((12, 64), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 64), torch.float32), ((64, 128), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128), torch.float32), ((128, 784), torch.float32)],
        {"model_names": ["pt_autoencoder_linear_img_enc_github"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 128, 1024), torch.float32), ((1024, 9), torch.float32)],
        {"model_names": ["pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 1024), torch.float32), ((1024, 4096), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 256, 1024), torch.float32), ((1024, 51200), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 1920), torch.bfloat16), ((1920, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_densenet_densenet201_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((1, 1408), torch.bfloat16), ((1408, 1000), torch.bfloat16)],
        {
            "model_names": ["pt_efficientnet_efficientnet_b2_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Matmul2,
        [((256, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 256, 64), torch.float32), ((12, 64, 256), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 256, 256), torch.float32), ((12, 256, 64), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (Matmul26, [((256, 768), torch.float32)], {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Matmul27, [((256, 768), torch.float32)], {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99}),
    (Matmul28, [((256, 3072), torch.float32)], {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99}),
    (
        Matmul2,
        [((1, 256, 768), torch.float32), ((768, 50257), torch.float32)],
        {"model_names": ["pt_gpt_gpt2_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((256, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 256, 768), torch.float32), ((768, 50272), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 768), torch.float32), ((768, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 32, 64), torch.float32), ((12, 64, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 32, 32), torch.float32), ((12, 32, 64), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 768), torch.float32), ((768, 3072), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 3072), torch.float32), ((3072, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 768), torch.float32), ((768, 2), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 2048), torch.float32), ((2048, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (Matmul42, [((1, 16, 1), torch.float32)], {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99}),
    (
        Matmul2,
        [((32, 12, 64), torch.float32), ((32, 64, 12), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((32, 12, 12), torch.float32), ((32, 12, 64), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((12, 2048), torch.float32), ((2048, 8192), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 12, 8192), torch.float32), ((8192, 2048), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 12, 2048), torch.float32), ((2048, 2), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((513, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((8, 513, 64), torch.float32), ((8, 64, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((8, 513, 513), torch.float32), ((8, 513, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((61, 512), torch.float32), ((512, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((8, 61, 64), torch.float32), ((8, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((8, 61, 61), torch.float32), ((8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 61, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 61, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((8, 513, 64), torch.float32), ((8, 64, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((8, 513, 61), torch.float32), ((8, 61, 64), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 513, 512), torch.float32), ((512, 2048), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 513, 2048), torch.float32), ((2048, 512), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
    (
        Matmul2,
        [((1, 513, 512), torch.float32), ((512, 32128), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Matmul")

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
