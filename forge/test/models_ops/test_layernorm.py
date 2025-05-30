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


class Layernorm0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm0_const_1", shape=(768,), dtype=torch.float32)
        self.add_constant("layernorm0_const_2", shape=(768,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm0_const_1"),
            self.get_constant("layernorm0_const_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm1_const_1", shape=(256,), dtype=torch.float32)
        self.add_constant("layernorm1_const_2", shape=(256,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm1_const_1"),
            self.get_constant("layernorm1_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm2_const_1", shape=(64,), dtype=torch.float32)
        self.add_constant("layernorm2_const_2", shape=(64,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm2_const_1"),
            self.get_constant("layernorm2_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm3_const_1", shape=(128,), dtype=torch.float32)
        self.add_constant("layernorm3_const_2", shape=(128,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm3_const_1"),
            self.get_constant("layernorm3_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm4_const_1", shape=(320,), dtype=torch.float32)
        self.add_constant("layernorm4_const_2", shape=(320,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm4_const_1"),
            self.get_constant("layernorm4_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm5_const_1", shape=(512,), dtype=torch.float32)
        self.add_constant("layernorm5_const_2", shape=(512,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm5_const_1"),
            self.get_constant("layernorm5_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm6.weight_1",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm6.weight_2",
            forge.Parameter(*(128,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm6.weight_1"),
            self.get_parameter("layernorm6.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm7.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm7.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm7.weight_1"),
            self.get_parameter("layernorm7.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm8.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm8.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm8.weight_1"),
            self.get_parameter("layernorm8.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm9.weight_1",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm9.weight_2",
            forge.Parameter(*(4096,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm9.weight_1"),
            self.get_parameter("layernorm9.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm10.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm10.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm10.weight_1"),
            self.get_parameter("layernorm10.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm11.weight_1",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm11.weight_2",
            forge.Parameter(*(1024,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm11.weight_1"),
            self.get_parameter("layernorm11.weight_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm12.weight_1",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm12.weight_2",
            forge.Parameter(*(2048,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm12.weight_1"),
            self.get_parameter("layernorm12.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm13.weight_1",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm13.weight_2",
            forge.Parameter(*(768,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm13.weight_1"),
            self.get_parameter("layernorm13.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm14.weight_1",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm14.weight_2",
            forge.Parameter(*(2560,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm14.weight_1"),
            self.get_parameter("layernorm14.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm15_const_1", shape=(1024,), dtype=torch.float32)
        self.add_constant("layernorm15_const_2", shape=(1024,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm15_const_1"),
            self.get_constant("layernorm15_const_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm16_const_1", shape=(384,), dtype=torch.float32)
        self.add_constant("layernorm16_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm16_const_1"),
            self.get_constant("layernorm16_const_2"),
            dim=-1,
            epsilon=0.0,
        )
        return layernorm_output_1


class Layernorm17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm17.weight_1",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm17.weight_2",
            forge.Parameter(*(1536,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm17.weight_1"),
            self.get_parameter("layernorm17.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm18(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm18.weight_1",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm18.weight_2",
            forge.Parameter(*(512,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm18.weight_1"),
            self.get_parameter("layernorm18.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm19(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm19.weight_1",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm19.weight_2",
            forge.Parameter(*(384,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm19.weight_1"),
            self.get_parameter("layernorm19.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm20(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm20_const_1", shape=(32,), dtype=torch.float32)
        self.add_constant("layernorm20_const_2", shape=(32,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm20_const_1"),
            self.get_constant("layernorm20_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm21(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm21_const_1", shape=(160,), dtype=torch.float32)
        self.add_constant("layernorm21_const_2", shape=(160,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm21_const_1"),
            self.get_constant("layernorm21_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm22(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "layernorm22.weight_1",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )
        self.add_parameter(
            "layernorm22.weight_2",
            forge.Parameter(*(1280,), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_parameter("layernorm22.weight_1"),
            self.get_parameter("layernorm22.weight_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm23(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm23_const_1", shape=(96,), dtype=torch.float32)
        self.add_constant("layernorm23_const_2", shape=(96,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm23_const_1"),
            self.get_constant("layernorm23_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm24(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm24_const_1", shape=(192,), dtype=torch.float32)
        self.add_constant("layernorm24_const_2", shape=(192,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm24_const_1"),
            self.get_constant("layernorm24_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm25(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm25_const_1", shape=(384,), dtype=torch.float32)
        self.add_constant("layernorm25_const_2", shape=(384,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm25_const_1"),
            self.get_constant("layernorm25_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm26(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm26_const_1", shape=(768,), dtype=torch.float32)
        self.add_constant("layernorm26_const_2", shape=(768,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm26_const_1"),
            self.get_constant("layernorm26_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


class Layernorm27(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("layernorm27_const_1", shape=(1280,), dtype=torch.float32)
        self.add_constant("layernorm27_const_2", shape=(1280,), dtype=torch.float32)

    def forward(self, layernorm_input_0):
        layernorm_output_1 = forge.op.Layernorm(
            "",
            layernorm_input_0,
            self.get_constant("layernorm27_const_1"),
            self.get_constant("layernorm27_const_2"),
            dim=-1,
            epsilon=1e-05,
        )
        return layernorm_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Layernorm0,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm1,
        [((1, 100, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 280, 256), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm2,
        [((1, 16384, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm2,
        [((1, 256, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 4096, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm3,
        [((1, 256, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 1024, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm4,
        [((1, 256, 320), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm5,
        [((1, 256, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 128, 128), torch.float32)],
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
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 128, 768), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_distilbert_distilbert_base_uncased_mlm_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_albert_base_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder",
                "pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
                "pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm8,
        [((1, 128, 2048), torch.float32)],
        {
            "model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm9,
        [((1, 128, 4096), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm10,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bart_facebook_bart_large_mnli_seq_cls_hf",
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm11,
        [((1, 128, 1024), torch.float32)],
        {
            "model_names": [
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 6, 768), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm12,
        [((1, 32, 2048), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm13,
        [((1, 32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm13,
        [((32, 768), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm14,
        [((1, 256, 2560), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm7,
        [((1, 204, 768), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "0.0"}},
    ),
    (
        Layernorm13,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm13,
        [((1, 1500, 768), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm15,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm16,
        [((1, 13, 384), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm17,
        [((1, 32, 1536), torch.float32)],
        {
            "model_names": ["pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((2, 7, 512), torch.float32)],
        {
            "model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm7,
        [((1, 384, 768), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm13,
        [((1, 7, 768), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm12,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_1_3b_clm_hf", "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm12,
        [((256, 2048), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm10,
        [((1, 32, 1024), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "1e-05"}},
    ),
    (
        Layernorm12,
        [((1, 7, 2048), torch.float32)],
        {
            "model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm7,
        [((1, 201, 768), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm17,
        [((1, 1536), torch.float32)],
        {
            "model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 1, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm19,
        [((1, 1500, 384), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 16384, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm20,
        [((1, 256, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm2,
        [((1, 4096, 64), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm21,
        [((1, 1024, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm21,
        [((1, 256, 160), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm1,
        [((1, 256, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm0,
        [((1, 197, 768), torch.float32)],
        {
            "model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm22,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm22,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm0,
        [((1, 128, 768), torch.float32)],
        {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99, "args": {"dim": "-1", "epsilon": "0.0"}},
    ),
    (
        Layernorm23,
        [((1, 4096, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm24,
        [((1, 1024, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm25,
        [((1, 256, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm26,
        [((1, 64, 768), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((1, 2, 1280), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm27,
        [((1, 1500, 1280), torch.float32)],
        {
            "model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm6,
        [((1, 14, 128), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm7,
        [((1, 14, 768), torch.float32)],
        {
            "model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm11,
        [((1, 384, 1024), torch.float32)],
        {
            "model_names": ["pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "0.0"},
        },
    ),
    (
        Layernorm17,
        [((2, 1, 1536), torch.float32)],
        {
            "model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
    (
        Layernorm18,
        [((1, 1500, 512), torch.float32)],
        {
            "model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1", "epsilon": "1e-05"},
        },
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Layernorm")

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
