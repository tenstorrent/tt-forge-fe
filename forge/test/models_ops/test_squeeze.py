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


class Squeeze0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-2)
        return squeeze_output_1


class Squeeze1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=1)
        return squeeze_output_1


class Squeeze2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-3)
        return squeeze_output_1


class Squeeze3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-1)
        return squeeze_output_1


class Squeeze4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=-5)
        return squeeze_output_1


class Squeeze5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, squeeze_input_0):
        squeeze_output_1 = forge.op.Squeeze("", squeeze_input_0, dim=3)
        return squeeze_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Squeeze0,
        [((1, 16384, 1, 64), torch.float32)],
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
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 1, 64), torch.float32)],
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
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b4_img_cls_hf",
                "onnx_segformer_nvidia_mit_b5_img_cls_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "1"},
        },
    ),
    (
        Squeeze2,
        [((1, 1, 11), torch.float32)],
        {"model_names": ["pd_bert_chinese_roberta_base_qa_padlenlp"], "pcc": 0.99, "args": {"dim": "-3"}},
    ),
    (
        Squeeze3,
        [((1, 120, 12, 1), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze4,
        [((1, 1, 8, 12, 15), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-5"},
        },
    ),
    (
        Squeeze0,
        [((1, 120, 1, 12), torch.float32)],
        {
            "model_names": ["pd_paddleocr_v4_rec_en_scene_text_recognition_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 2048, 1, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze3,
        [((1, 2048, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_101_img_cls_paddlemodels", "pd_resnet_152_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 128, 768, 1), torch.float32)],
        {
            "model_names": [
                "pt_albert_base_v1_token_cls_hf",
                "pt_albert_base_v2_mlm_hf",
                "pt_albert_base_v2_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 128, 2048, 1), torch.float32)],
        {
            "model_names": ["pt_albert_xlarge_v2_token_cls_hf", "pt_albert_xlarge_v1_mlm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 128, 4096, 1), torch.float32)],
        {
            "model_names": [
                "pt_albert_xxlarge_v1_mlm_hf",
                "pt_albert_xxlarge_v1_token_cls_hf",
                "pt_albert_xxlarge_v2_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 256, 16, 32, 1), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 128, 1), torch.float32)],
        {"model_names": ["pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze3,
        [((1, 32, 1), torch.float32)],
        {
            "model_names": ["pt_opt_facebook_opt_125m_qa_hf", "pt_opt_facebook_opt_350m_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze3,
        [((1, 384, 1), torch.float32)],
        {
            "model_names": [
                "onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
                "pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf",
                "pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1, 768), torch.float32)],
        {
            "model_names": [
                "pd_bert_bert_base_uncased_seq_cls_padlenlp",
                "pd_bert_bert_base_japanese_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_seq_cls_padlenlp",
                "pd_chineseclip_text_ofa_sys_chinese_clip_vit_base_patch16_text_enc_padlenlp",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 1, 9), torch.float32)],
        {
            "model_names": ["pd_ernie_1_0_qa_padlenlp", "pd_bert_bert_base_uncased_qa_padlenlp"],
            "pcc": 0.99,
            "args": {"dim": "-3"},
        },
    ),
    (
        Squeeze3,
        [((1, 128, 1024, 1), torch.float32)],
        {
            "model_names": ["pt_albert_large_v1_mlm_hf", "pt_albert_large_v2_token_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze5,
        [((1, 100, 8, 1, 280), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Squeeze0,
        [((1, 16384, 1, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze0,
        [((1, 256, 1, 32), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze1,
        [((1, 1, 256), torch.float32)],
        {"model_names": ["onnx_segformer_nvidia_mit_b0_img_cls_hf"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Squeeze0,
        [((1, 288, 1, 25), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze2,
        [((1, 1, 288), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3"},
        },
    ),
    (
        Squeeze2,
        [((1, 1, 96), torch.float32)],
        {
            "model_names": [
                "pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels",
                "pd_paddleocr_v0_rec_en_scene_text_recognition_paddlemodels",
            ],
            "pcc": 0.99,
            "args": {"dim": "-3"},
        },
    ),
    (
        Squeeze0,
        [((1, 512, 1, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze3,
        [((1, 512, 1), torch.float32)],
        {
            "model_names": ["pd_resnet_18_img_cls_paddlemodels", "pd_resnet_34_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 9216, 1, 1), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "-2"}},
    ),
    (
        Squeeze3,
        [((1, 9216, 1), torch.float32)],
        {"model_names": ["pd_alexnet_base_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 1024, 1, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-2"},
        },
    ),
    (
        Squeeze3,
        [((1, 1024, 1), torch.float32)],
        {
            "model_names": ["pd_densenet_121_img_cls_paddlemodels", "pd_mobilenetv1_basic_img_cls_paddlemodels"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Squeeze0,
        [((1, 1280, 1, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "-2"}},
    ),
    (
        Squeeze3,
        [((1, 1280, 1), torch.float32)],
        {"model_names": ["pd_mobilenetv2_basic_img_cls_paddlemodels"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze3,
        [((1, 14, 768, 1), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze3,
        [((1, 14, 1), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Squeeze0,
        [((1, 2048, 1, 9), torch.float32)],
        {"model_names": ["pt_mamba_state_spaces_mamba_370m_hf_clm_hf"], "pcc": 0.99, "args": {"dim": "-2"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Squeeze")

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
