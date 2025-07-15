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


class Cast0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.float32)
        return cast_output_1


class Cast1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int64)
        return cast_output_1


class Cast2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.int32)
        return cast_output_1


class Cast3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, cast_input_0):
        cast_output_1 = forge.op.Cast("", cast_input_0, dtype=torch.bool)
        return cast_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Cast0,
        [((1, 1, 1, 11), torch.int64)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 128, 128), torch.int64)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 128, 128), torch.bool)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 256, 256), torch.int64)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 256, 256), torch.bool)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast1,
        [((1, 9), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.int64"}},
    ),
    (
        Cast2,
        [((1, 9), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
    (
        Cast0,
        [((1, 9), torch.bool)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast3,
        [((1, 9), torch.int64)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 9), torch.int32)],
        {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 256, 6, 20), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 256, 12, 40), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 128, 12, 40), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 128, 24, 80), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 64, 24, 80), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 64, 48, 160), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 32, 48, 160), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 32, 96, 320), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 16, 96, 320), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast3,
        [((1, 16, 192, 640), torch.bool)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "args": {"dtype": "torch.bool"},
        },
    ),
    (
        Cast0,
        [((1, 1, 6, 6), torch.int64)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 6, 6), torch.bool)],
        {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99, "args": {"dtype": "torch.float32"}},
    ),
    (
        Cast0,
        [((1, 1, 35, 35), torch.int64)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 35, 35), torch.bool)],
        {
            "model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 29, 29), torch.int64)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast0,
        [((1, 1, 29, 29), torch.bool)],
        {
            "model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast3,
        [((1, 32, 480, 640), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 64, 240, 320), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 32, 240, 320), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 128, 120, 160), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 64, 120, 160), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 256, 60, 80), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 128, 60, 80), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 512, 30, 40), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 256, 30, 40), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 1024, 15, 20), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast3,
        [((1, 512, 15, 20), torch.bool)],
        {"model_names": ["pt_yolo_v4_default_obj_det_github"], "pcc": 0.99, "args": {"dtype": "torch.bool"}},
    ),
    (
        Cast0,
        [((1, 1, 1, 9), torch.int64)],
        {
            "model_names": ["pd_bert_chinese_roberta_base_mlm_padlenlp", "pd_bert_bert_base_uncased_mlm_padlenlp"],
            "pcc": 0.99,
            "args": {"dtype": "torch.float32"},
        },
    ),
    (
        Cast2,
        [((1, 32), torch.bool)],
        {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99, "args": {"dtype": "torch.int32"}},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Cast")

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
