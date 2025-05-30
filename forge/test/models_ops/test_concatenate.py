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


class Concatenate0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, concatenate_input_0, concatenate_input_1, concatenate_input_2, concatenate_input_3):
        concatenate_output_1 = forge.op.Concatenate(
            "", concatenate_input_0, concatenate_input_1, concatenate_input_2, concatenate_input_3, axis=-3
        )
        return concatenate_output_1


class Concatenate1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, concatenate_input_0, concatenate_input_1):
        concatenate_output_1 = forge.op.Concatenate("", concatenate_input_0, concatenate_input_1, axis=-3)
        return concatenate_output_1


class Concatenate2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, concatenate_input_0, concatenate_input_1, concatenate_input_2):
        concatenate_output_1 = forge.op.Concatenate(
            "", concatenate_input_0, concatenate_input_1, concatenate_input_2, axis=-3
        )
        return concatenate_output_1


class Concatenate3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(
        self, concatenate_input_0, concatenate_input_1, concatenate_input_2, concatenate_input_3, concatenate_input_4
    ):
        concatenate_output_1 = forge.op.Concatenate(
            "",
            concatenate_input_0,
            concatenate_input_1,
            concatenate_input_2,
            concatenate_input_3,
            concatenate_input_4,
            axis=-3,
        )
        return concatenate_output_1


class Concatenate4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(
        self,
        concatenate_input_0,
        concatenate_input_1,
        concatenate_input_2,
        concatenate_input_3,
        concatenate_input_4,
        concatenate_input_5,
    ):
        concatenate_output_1 = forge.op.Concatenate(
            "",
            concatenate_input_0,
            concatenate_input_1,
            concatenate_input_2,
            concatenate_input_3,
            concatenate_input_4,
            concatenate_input_5,
            axis=-3,
        )
        return concatenate_output_1


class Concatenate5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, concatenate_input_0, concatenate_input_1, concatenate_input_2, concatenate_input_3):
        concatenate_output_1 = forge.op.Concatenate(
            "", concatenate_input_0, concatenate_input_1, concatenate_input_2, concatenate_input_3, axis=-2
        )
        return concatenate_output_1


class Concatenate6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, concatenate_input_0, concatenate_input_1):
        concatenate_output_1 = forge.op.Concatenate("", concatenate_input_0, concatenate_input_1, axis=-1)
        return concatenate_output_1


class Concatenate7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(
        self,
        concatenate_input_0,
        concatenate_input_1,
        concatenate_input_2,
        concatenate_input_3,
        concatenate_input_4,
        concatenate_input_5,
        concatenate_input_6,
    ):
        concatenate_output_1 = forge.op.Concatenate(
            "",
            concatenate_input_0,
            concatenate_input_1,
            concatenate_input_2,
            concatenate_input_3,
            concatenate_input_4,
            concatenate_input_5,
            concatenate_input_6,
            axis=-3,
        )
        return concatenate_output_1


class Concatenate8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_parameter(
            "concatenate8.weight_0",
            forge.Parameter(*(1, 1, 768), requires_grad=True, dev_data_format=forge.DataFormat.Float32),
        )

    def forward(self, concatenate_input_1):
        concatenate_output_1 = forge.op.Concatenate(
            "", self.get_parameter("concatenate8.weight_0"), concatenate_input_1, axis=-2
        )
        return concatenate_output_1


class Concatenate9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, concatenate_input_0, concatenate_input_1, concatenate_input_2):
        concatenate_output_1 = forge.op.Concatenate(
            "", concatenate_input_0, concatenate_input_1, concatenate_input_2, axis=-1
        )
        return concatenate_output_1


class Concatenate10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, concatenate_input_0, concatenate_input_1):
        concatenate_output_1 = forge.op.Concatenate("", concatenate_input_0, concatenate_input_1, axis=-2)
        return concatenate_output_1


class Concatenate11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(
        self,
        concatenate_input_0,
        concatenate_input_1,
        concatenate_input_2,
        concatenate_input_3,
        concatenate_input_4,
        concatenate_input_5,
        concatenate_input_6,
        concatenate_input_7,
        concatenate_input_8,
        concatenate_input_9,
        concatenate_input_10,
        concatenate_input_11,
        concatenate_input_12,
        concatenate_input_13,
        concatenate_input_14,
        concatenate_input_15,
        concatenate_input_16,
        concatenate_input_17,
        concatenate_input_18,
        concatenate_input_19,
        concatenate_input_20,
        concatenate_input_21,
        concatenate_input_22,
        concatenate_input_23,
        concatenate_input_24,
    ):
        concatenate_output_1 = forge.op.Concatenate(
            "",
            concatenate_input_0,
            concatenate_input_1,
            concatenate_input_2,
            concatenate_input_3,
            concatenate_input_4,
            concatenate_input_5,
            concatenate_input_6,
            concatenate_input_7,
            concatenate_input_8,
            concatenate_input_9,
            concatenate_input_10,
            concatenate_input_11,
            concatenate_input_12,
            concatenate_input_13,
            concatenate_input_14,
            concatenate_input_15,
            concatenate_input_16,
            concatenate_input_17,
            concatenate_input_18,
            concatenate_input_19,
            concatenate_input_20,
            concatenate_input_21,
            concatenate_input_22,
            concatenate_input_23,
            concatenate_input_24,
            axis=-3,
        )
        return concatenate_output_1


class Concatenate12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, concatenate_input_0, concatenate_input_1, concatenate_input_2, concatenate_input_3):
        concatenate_output_1 = forge.op.Concatenate(
            "", concatenate_input_0, concatenate_input_1, concatenate_input_2, concatenate_input_3, axis=-1
        )
        return concatenate_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Concatenate0,
        [
            ((1, 24, 120, 120), torch.float32),
            ((1, 24, 120, 120), torch.float32),
            ((1, 24, 120, 120), torch.float32),
            ((1, 24, 120, 120), torch.float32),
        ],
        {"model_names": ["TranslatedLayer"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 128, 56, 56), torch.float32), ((1, 128, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate1,
        [((1, 256, 28, 28), torch.float32), ((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate2,
        [((1, 256, 28, 28), torch.float32), ((1, 256, 28, 28), torch.float32), ((1, 256, 28, 28), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate3,
        [
            ((1, 256, 28, 28), torch.float32),
            ((1, 256, 28, 28), torch.float32),
            ((1, 128, 28, 28), torch.float32),
            ((1, 256, 28, 28), torch.float32),
            ((1, 256, 28, 28), torch.float32),
        ],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate1,
        [((1, 512, 14, 14), torch.float32), ((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate2,
        [((1, 512, 14, 14), torch.float32), ((1, 512, 14, 14), torch.float32), ((1, 512, 14, 14), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate0,
        [
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
        ],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate4,
        [
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 256, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
        ],
        {
            "model_names": ["onnx_dla_dla102x2_visual_bb_torchvision", "onnx_dla_dla102x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate2,
        [((1, 1024, 7, 7), torch.float32), ((1, 1024, 7, 7), torch.float32), ((1, 512, 7, 7), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla102x2_visual_bb_torchvision",
                "onnx_dla_dla102x_visual_bb_torchvision",
                "onnx_dla_dla60_visual_bb_torchvision",
                "onnx_dla_dla169_visual_bb_torchvision",
                "onnx_dla_dla60x_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate1,
        [((1, 64, 56, 56), torch.float32), ((1, 64, 56, 56), torch.float32)],
        {
            "model_names": [
                "onnx_dla_dla46x_c_visual_bb_torchvision",
                "onnx_dla_dla34_visual_bb_torchvision",
                "onnx_dla_dla60x_c_visual_bb_torchvision",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate1,
        [((1, 64, 28, 28), torch.float32), ((1, 64, 28, 28), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate0,
        [
            ((1, 64, 28, 28), torch.float32),
            ((1, 64, 28, 28), torch.float32),
            ((1, 64, 28, 28), torch.float32),
            ((1, 64, 28, 28), torch.float32),
        ],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate1,
        [((1, 128, 14, 14), torch.float32), ((1, 128, 14, 14), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate0,
        [
            ((1, 128, 14, 14), torch.float32),
            ((1, 128, 14, 14), torch.float32),
            ((1, 64, 14, 14), torch.float32),
            ((1, 128, 14, 14), torch.float32),
        ],
        {"model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate2,
        [((1, 256, 7, 7), torch.float32), ((1, 256, 7, 7), torch.float32), ((1, 128, 7, 7), torch.float32)],
        {
            "model_names": ["onnx_dla_dla46x_c_visual_bb_torchvision", "onnx_dla_dla60x_c_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate0,
        [
            ((1, 768, 128, 128), torch.float32),
            ((1, 768, 128, 128), torch.float32),
            ((1, 768, 128, 128), torch.float32),
            ((1, 768, 128, 128), torch.float32),
        ],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate4,
        [
            ((1, 128, 56, 56), torch.float32),
            ((1, 64, 56, 56), torch.float32),
            ((1, 64, 56, 56), torch.float32),
            ((1, 64, 56, 56), torch.float32),
            ((1, 64, 56, 56), torch.float32),
            ((1, 64, 56, 56), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate4,
        [
            ((1, 128, 28, 28), torch.float32),
            ((1, 80, 28, 28), torch.float32),
            ((1, 80, 28, 28), torch.float32),
            ((1, 80, 28, 28), torch.float32),
            ((1, 80, 28, 28), torch.float32),
            ((1, 80, 28, 28), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate4,
        [
            ((1, 256, 14, 14), torch.float32),
            ((1, 96, 14, 14), torch.float32),
            ((1, 96, 14, 14), torch.float32),
            ((1, 96, 14, 14), torch.float32),
            ((1, 96, 14, 14), torch.float32),
            ((1, 96, 14, 14), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate4,
        [
            ((1, 384, 7, 7), torch.float32),
            ((1, 112, 7, 7), torch.float32),
            ((1, 112, 7, 7), torch.float32),
            ((1, 112, 7, 7), torch.float32),
            ((1, 112, 7, 7), torch.float32),
            ((1, 112, 7, 7), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet27s_obj_det_osmr"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 480, 1, 12), torch.float32), ((1, 480, 1, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate5,
        [
            ((256, 1024), torch.float32),
            ((256, 1024), torch.float32),
            ((256, 1024), torch.float32),
            ((256, 1024), torch.float32),
        ],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"axis": "-2"},
        },
    ),
    (
        Concatenate6,
        [((1, 256, 16, 32), torch.float32), ((1, 256, 16, 32), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"axis": "-1"},
        },
    ),
    (
        Concatenate6,
        [((1, 256, 32), torch.float32), ((1, 256, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 32, 256, 32), torch.float32), ((1, 32, 256, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 8, 256, 32), torch.float32), ((1, 8, 256, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 256, 16), torch.float32), ((1, 256, 16), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 32, 256, 16), torch.float32), ((1, 32, 256, 16), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 32, 256, 32), torch.float32), ((1, 32, 256, 48), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate0,
        [
            ((1, 256, 28, 28), torch.float32),
            ((1, 256, 28, 28), torch.float32),
            ((1, 128, 28, 28), torch.float32),
            ((1, 256, 28, 28), torch.float32),
        ],
        {
            "model_names": ["onnx_dla_dla60_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate3,
        [
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 256, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
        ],
        {
            "model_names": ["onnx_dla_dla60_visual_bb_torchvision", "onnx_dla_dla60x_visual_bb_torchvision"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate4,
        [
            ((1, 128, 56, 56), torch.float32),
            ((1, 128, 56, 56), torch.float32),
            ((1, 128, 56, 56), torch.float32),
            ((1, 128, 56, 56), torch.float32),
            ((1, 128, 56, 56), torch.float32),
            ((1, 128, 56, 56), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate4,
        [
            ((1, 256, 28, 28), torch.float32),
            ((1, 160, 28, 28), torch.float32),
            ((1, 160, 28, 28), torch.float32),
            ((1, 160, 28, 28), torch.float32),
            ((1, 160, 28, 28), torch.float32),
            ((1, 160, 28, 28), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate4,
        [
            ((1, 512, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate4,
        [
            ((1, 768, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
            ((1, 192, 14, 14), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate4,
        [
            ((1, 768, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate4,
        [
            ((1, 1024, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
            ((1, 224, 7, 7), torch.float32),
        ],
        {"model_names": ["onnx_vovnet_vovnet_v1_57_obj_det_torchhub"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate0,
        [
            ((1, 24, 112, 112), torch.float32),
            ((1, 24, 112, 112), torch.float32),
            ((1, 24, 112, 112), torch.float32),
            ((1, 24, 112, 112), torch.float32),
        ],
        {
            "model_names": ["pd_paddleocr_v4_det_ch_scene_text_detection_paddlemodels"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate6,
        [((1, 4, 32), torch.float32), ((1, 4, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 32, 4, 32), torch.float32), ((1, 32, 4, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 8, 4, 32), torch.float32), ((1, 8, 4, 32), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 7, 16), torch.float32), ((1, 7, 16), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 32, 7, 16), torch.float32), ((1, 32, 7, 16), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 32, 7, 32), torch.float32), ((1, 32, 7, 32), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate1,
        [((100, 256, 14, 20), torch.float32), ((100, 8, 14, 20), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate2,
        [((1, 128, 56, 56), torch.float32), ((1, 128, 56, 56), torch.float32), ((1, 128, 56, 56), torch.float32)],
        {"model_names": ["onnx_dla_dla169_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate3,
        [
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
        ],
        {"model_names": ["onnx_dla_dla169_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate7,
        [
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 256, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
            ((1, 512, 14, 14), torch.float32),
        ],
        {"model_names": ["onnx_dla_dla169_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate8,
        [((1, 196, 768), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99, "args": {"axis": "-2"}},
    ),
    (
        Concatenate2,
        [((1, 16, 160, 160), torch.float32), ((1, 16, 160, 160), torch.float32), ((1, 16, 160, 160), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate0,
        [
            ((1, 32, 80, 80), torch.float32),
            ((1, 32, 80, 80), torch.float32),
            ((1, 32, 80, 80), torch.float32),
            ((1, 32, 80, 80), torch.float32),
        ],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate0,
        [
            ((1, 64, 40, 40), torch.float32),
            ((1, 64, 40, 40), torch.float32),
            ((1, 64, 40, 40), torch.float32),
            ((1, 64, 40, 40), torch.float32),
        ],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate2,
        [((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32), ((1, 128, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate0,
        [
            ((1, 128, 20, 20), torch.float32),
            ((1, 128, 20, 20), torch.float32),
            ((1, 128, 20, 20), torch.float32),
            ((1, 128, 20, 20), torch.float32),
        ],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 256, 40, 40), torch.float32), ((1, 128, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate2,
        [((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32), ((1, 64, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 128, 80, 80), torch.float32), ((1, 64, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate2,
        [((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32), ((1, 32, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 64, 80, 80), torch.float32), ((1, 80, 80, 80), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 64, 40, 40), torch.float32), ((1, 128, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 64, 40, 40), torch.float32), ((1, 80, 40, 40), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 128, 20, 20), torch.float32), ((1, 256, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 64, 20, 20), torch.float32), ((1, 80, 20, 20), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate9,
        [((1, 144, 6400), torch.float32), ((1, 144, 1600), torch.float32), ((1, 144, 400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate10,
        [((1, 2, 8400), torch.float32), ((1, 2, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-2"}},
    ),
    (
        Concatenate10,
        [((1, 4, 8400), torch.float32), ((1, 80, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"axis": "-2"}},
    ),
    (
        Concatenate11,
        [
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
            ((1, 1, 48), torch.float32),
        ],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate1,
        [((25, 1, 1, 48), torch.float32), ((25, 1, 1, 48), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate1,
        [((1, 128, 28, 28), torch.float32), ((1, 128, 28, 28), torch.float32)],
        {"model_names": ["onnx_dla_dla34_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate0,
        [
            ((1, 128, 28, 28), torch.float32),
            ((1, 128, 28, 28), torch.float32),
            ((1, 64, 28, 28), torch.float32),
            ((1, 128, 28, 28), torch.float32),
        ],
        {"model_names": ["onnx_dla_dla34_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 256, 14, 14), torch.float32), ((1, 256, 14, 14), torch.float32)],
        {"model_names": ["onnx_dla_dla34_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate0,
        [
            ((1, 256, 14, 14), torch.float32),
            ((1, 256, 14, 14), torch.float32),
            ((1, 128, 14, 14), torch.float32),
            ((1, 256, 14, 14), torch.float32),
        ],
        {"model_names": ["onnx_dla_dla34_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate2,
        [((1, 512, 7, 7), torch.float32), ((1, 512, 7, 7), torch.float32), ((1, 256, 7, 7), torch.float32)],
        {"model_names": ["onnx_dla_dla34_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate2,
        [((1, 128, 14, 14), torch.float32), ((1, 128, 14, 14), torch.float32), ((1, 128, 14, 14), torch.float32)],
        {"model_names": ["onnx_dla_dla60x_c_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate3,
        [
            ((1, 128, 14, 14), torch.float32),
            ((1, 128, 14, 14), torch.float32),
            ((1, 64, 14, 14), torch.float32),
            ((1, 128, 14, 14), torch.float32),
            ((1, 128, 14, 14), torch.float32),
        ],
        {"model_names": ["onnx_dla_dla60x_c_visual_bb_torchvision"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate0,
        [
            ((1, 256, 128, 128), torch.float32),
            ((1, 256, 128, 128), torch.float32),
            ((1, 256, 128, 128), torch.float32),
            ((1, 256, 128, 128), torch.float32),
        ],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate1,
        [((1, 60, 64, 96), torch.float32), ((1, 4, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate10,
        [((1, 64, 60, 96), torch.float32), ((1, 64, 4, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-2"},
        },
    ),
    (
        Concatenate1,
        [((1, 4, 64, 96), torch.float32), ((1, 60, 64, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate10,
        [((1, 64, 4, 96), torch.float32), ((1, 64, 60, 96), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-2"},
        },
    ),
    (
        Concatenate12,
        [
            ((1, 32, 32, 96), torch.float32),
            ((1, 32, 32, 96), torch.float32),
            ((1, 32, 32, 96), torch.float32),
            ((1, 32, 32, 96), torch.float32),
        ],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-1"},
        },
    ),
    (
        Concatenate1,
        [((1, 28, 32, 192), torch.float32), ((1, 4, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate10,
        [((1, 32, 28, 192), torch.float32), ((1, 32, 4, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-2"},
        },
    ),
    (
        Concatenate1,
        [((1, 4, 32, 192), torch.float32), ((1, 28, 32, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate10,
        [((1, 32, 4, 192), torch.float32), ((1, 32, 28, 192), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-2"},
        },
    ),
    (
        Concatenate12,
        [
            ((1, 16, 16, 192), torch.float32),
            ((1, 16, 16, 192), torch.float32),
            ((1, 16, 16, 192), torch.float32),
            ((1, 16, 16, 192), torch.float32),
        ],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-1"},
        },
    ),
    (
        Concatenate1,
        [((1, 12, 16, 384), torch.float32), ((1, 4, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate10,
        [((1, 16, 12, 384), torch.float32), ((1, 16, 4, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-2"},
        },
    ),
    (
        Concatenate1,
        [((1, 4, 16, 384), torch.float32), ((1, 12, 16, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-3"},
        },
    ),
    (
        Concatenate10,
        [((1, 16, 4, 384), torch.float32), ((1, 16, 12, 384), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-2"},
        },
    ),
    (
        Concatenate12,
        [
            ((1, 8, 8, 384), torch.float32),
            ((1, 8, 8, 384), torch.float32),
            ((1, 8, 8, 384), torch.float32),
            ((1, 8, 8, 384), torch.float32),
        ],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"axis": "-1"},
        },
    ),
    (
        Concatenate1,
        [((1, 64, 56, 56), torch.float32), ((1, 32, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 96, 56, 56), torch.float32), ((1, 32, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 128, 56, 56), torch.float32), ((1, 32, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 160, 56, 56), torch.float32), ((1, 32, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 192, 56, 56), torch.float32), ((1, 32, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 224, 56, 56), torch.float32), ((1, 32, 56, 56), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 128, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 160, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 192, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 224, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 256, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 288, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 320, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 352, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 384, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 416, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 448, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 480, 28, 28), torch.float32), ((1, 32, 28, 28), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 256, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 288, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 320, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 352, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 384, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 416, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 448, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 480, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 512, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 544, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 576, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 608, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 640, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 672, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 704, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 736, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 768, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 800, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 832, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 864, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 896, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 928, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 960, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 992, 14, 14), torch.float32), ((1, 32, 14, 14), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 512, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 544, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 576, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 608, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 640, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 672, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 704, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 736, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 768, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 800, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 832, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 864, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 896, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 928, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 960, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate1,
        [((1, 992, 7, 7), torch.float32), ((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pd_densenet_121_img_cls_paddlemodels"], "pcc": 0.99, "args": {"axis": "-3"}},
    ),
    (
        Concatenate6,
        [((1, 588, 64), torch.float32), ((1, 588, 64), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
    (
        Concatenate6,
        [((1, 16, 588, 64), torch.float32), ((1, 16, 588, 64), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"axis": "-1"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Concatenate")

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
