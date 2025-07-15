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


class Softmax0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, softmax_input_0):
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=-1)
        return softmax_output_1


class Softmax1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, softmax_input_0):
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=1)
        return softmax_output_1


class Softmax2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, softmax_input_0):
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=3)
        return softmax_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Softmax0,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 12, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 32, 4, 4), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 32, 256, 256), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 32, 5, 5), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 1, 16384, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 2, 4096, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 5, 1024, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 8, 256, 256), torch.bfloat16)],
        {
            "model_names": [
                "pt_segformer_nvidia_mit_b0_img_cls_hf",
                "pt_segformer_nvidia_mit_b1_img_cls_hf",
                "pt_segformer_nvidia_mit_b3_img_cls_hf",
                "pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((64, 3, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((16, 6, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((4, 12, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 24, 49, 49), torch.bfloat16)],
        {
            "model_names": ["pt_swin_swin_s_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 12, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_vit_google_vit_base_patch16_224_img_cls_hf",
                "pt_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "pt_vit_vit_b_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 16, 50, 50), torch.bfloat16)],
        {
            "model_names": ["pt_vit_vit_l_32_img_cls_torchvision"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax1,
        [((1, 16, 4, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "1"},
        },
    ),
    (
        Softmax2,
        [((1, 1, 16384, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax2,
        [((1, 2, 4096, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax2,
        [((1, 5, 1024, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax2,
        [((1, 8, 256, 256), torch.float32)],
        {
            "model_names": ["onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax0,
        [((1, 16, 197, 197), torch.bfloat16)],
        {
            "model_names": [
                "pt_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "pt_vit_vit_l_16_img_cls_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 32, 11, 11), torch.float32)],
        {"model_names": ["pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 16, 6, 6), torch.float32)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 12, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 12, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((64, 3, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((16, 6, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((4, 12, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 24, 64, 64), torch.float32)],
        {"model_names": ["pt_swin_swin_v2_t_img_cls_torchvision"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 12, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 12, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 12, 513, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_base_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 3, 197, 197), torch.bfloat16)],
        {
            "model_names": ["pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 14, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((16, 256, 256), torch.float32)],
        {"model_names": ["pt_xglm_facebook_xglm_564m_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 17, 4, 4480), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "1"},
        },
    ),
    (
        Softmax1,
        [((1, 17, 4, 1120), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "1"},
        },
    ),
    (
        Softmax1,
        [((1, 17, 4, 280), torch.bfloat16)],
        {
            "model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
            "args": {"dim": "1"},
        },
    ),
    (
        Softmax0,
        [((1, 1000), torch.float32)],
        {"model_names": ["tf_resnet_resnet50_img_cls_keras"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax2,
        [((1, 3, 197, 197), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Softmax1,
        [((1, 16, 4, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Softmax0,
        [((1, 16, 256, 256), torch.float32)],
        {"model_names": ["pt_codegen_salesforce_codegen_350m_nl_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 12, 256, 256), torch.float32)],
        {
            "model_names": ["pt_gpt_gpt2_text_gen_hf", "pt_opt_facebook_opt_125m_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 12, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 32, 12, 12), torch.float32)],
        {"model_names": ["pt_phi1_microsoft_phi_1_token_cls_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 8, 513, 513), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 8, 61, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 8, 513, 61), torch.float32)],
        {"model_names": ["pt_t5_t5_small_text_gen_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Softmax")

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
