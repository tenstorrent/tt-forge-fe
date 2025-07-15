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


class Subtract0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, subtract_input_0, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, subtract_input_1)
        return subtract_output_1


class Subtract1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract1_const_0", shape=(1,), dtype=torch.float32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract1_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract2_const_0", shape=(1, 2, 8400), dtype=torch.bfloat16)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract2_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract3_const_0", shape=(1,), dtype=torch.int32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract3_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract4(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract4_const_1", shape=(1,), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract4_const_1"))
        return subtract_output_1


class Subtract5(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract5_const_1", shape=(1, 256, 6, 20), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract5_const_1"))
        return subtract_output_1


class Subtract6(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract6_const_1", shape=(1, 256, 12, 40), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract6_const_1"))
        return subtract_output_1


class Subtract7(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract7_const_1", shape=(1, 128, 12, 40), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract7_const_1"))
        return subtract_output_1


class Subtract8(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract8_const_1", shape=(1, 128, 24, 80), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract8_const_1"))
        return subtract_output_1


class Subtract9(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract9_const_1", shape=(1, 64, 24, 80), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract9_const_1"))
        return subtract_output_1


class Subtract10(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract10_const_1", shape=(1, 64, 48, 160), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract10_const_1"))
        return subtract_output_1


class Subtract11(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract11_const_1", shape=(1, 32, 48, 160), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract11_const_1"))
        return subtract_output_1


class Subtract12(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract12_const_1", shape=(1, 32, 96, 320), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract12_const_1"))
        return subtract_output_1


class Subtract13(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract13_const_1", shape=(1, 16, 96, 320), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract13_const_1"))
        return subtract_output_1


class Subtract14(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract14_const_1", shape=(1, 16, 192, 640), dtype=torch.bfloat16)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract14_const_1"))
        return subtract_output_1


class Subtract15(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract15_const_0", shape=(5880, 2), dtype=torch.bfloat16)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract15_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract16(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract16_const_0", shape=(1, 2, 8400), dtype=torch.float32)

    def forward(self, subtract_input_1):
        subtract_output_1 = forge.op.Subtract("", self.get_constant("subtract16_const_0"), subtract_input_1)
        return subtract_output_1


class Subtract17(ForgeModule):
    def __init__(self, name):
        super().__init__(name)
        self.add_constant("subtract17_const_1", shape=(1,), dtype=torch.int64)

    def forward(self, subtract_input_0):
        subtract_output_1 = forge.op.Subtract("", subtract_input_0, self.get_constant("subtract17_const_1"))
        return subtract_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Subtract0,
        [((1, 11, 128), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 12, 11, 11), torch.float32), ((1, 12, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 11, 312), torch.float32), ((1, 11, 1), torch.float32)],
        {"model_names": ["pd_albert_chinese_tiny_mlm_padlenlp"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 1, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_large_v2_token_cls_hf",
                "pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader",
                "pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder",
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_distilbert_distilbert_base_cased_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract1,
        [((1, 1, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_phi1_microsoft_phi_1_clm_hf",
                "pt_xglm_facebook_xglm_564m_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Subtract2,
        [((1, 2, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract0,
        [((1, 2, 8400), torch.bfloat16), ((1, 2, 8400), torch.bfloat16)],
        {
            "model_names": ["pt_yolov8_yolov8x_obj_det_github", "pt_yolov9_default_obj_det_github"],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract0,
        [((1, 9, 768), torch.float32), ((1, 9, 1), torch.float32)],
        {
            "model_names": [
                "pd_roberta_rbt4_ch_seq_cls_padlenlp",
                "pd_bert_chinese_roberta_base_mlm_padlenlp",
                "pd_bert_bert_base_uncased_mlm_padlenlp",
            ],
            "pcc": 0.99,
        },
    ),
    (Subtract3, [((1, 9), torch.int32)], {"model_names": ["pd_roberta_rbt4_ch_seq_cls_padlenlp"], "pcc": 0.99}),
    (
        Subtract0,
        [((1, 12, 9, 9), torch.float32), ((1, 12, 9, 1), torch.float32)],
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
        Subtract4,
        [((1, 3, 192, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract5,
        [((1, 256, 6, 20), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract6,
        [((1, 256, 12, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract7,
        [((1, 128, 12, 40), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract8,
        [((1, 128, 24, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract9,
        [((1, 64, 24, 80), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract10,
        [((1, 64, 48, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract11,
        [((1, 32, 48, 160), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract12,
        [((1, 32, 96, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract13,
        [((1, 16, 96, 320), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (
        Subtract14,
        [((1, 16, 192, 640), torch.bfloat16)],
        {
            "model_names": [
                "pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_mono_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision",
                "pt_monodepth2_stereo_640x192_depth_prediction_torchvision",
            ],
            "pcc": 0.99,
            "default_df_override": "Float16_b",
        },
    ),
    (Subtract1, [((1, 1, 6, 6), torch.float32)], {"model_names": ["pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf"], "pcc": 0.99}),
    (
        Subtract1,
        [((1, 1, 35, 35), torch.float32)],
        {"model_names": ["pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Subtract1,
        [((1, 1, 29, 29), torch.float32)],
        {"model_names": ["pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf", "pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf"], "pcc": 0.99},
    ),
    (
        Subtract15,
        [((1, 5880, 2), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (
        Subtract0,
        [((1, 5880, 2), torch.bfloat16), ((1, 5880, 2), torch.bfloat16)],
        {"model_names": ["pt_yolo_v6_yolov6l_obj_det_torchhub"], "pcc": 0.99, "default_df_override": "Float16_b"},
    ),
    (Subtract16, [((1, 2, 8400), torch.float32)], {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99}),
    (
        Subtract0,
        [((1, 2, 8400), torch.float32), ((1, 2, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99},
    ),
    (
        Subtract0,
        [((1, 25, 6625), torch.float32), ((1, 25, 1), torch.float32)],
        {"model_names": ["pd_paddleocr_v0_rec_ch_scene_text_recognition_paddlemodels"], "pcc": 0.99},
    ),
    (Subtract17, [((1, 256), torch.int64)], {"model_names": ["pt_opt_facebook_opt_125m_clm_hf"], "pcc": 0.99}),
    (Subtract17, [((1, 32), torch.int64)], {"model_names": ["pt_opt_facebook_opt_125m_seq_cls_hf"], "pcc": 0.99}),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
def test_module(forge_module_and_shapes_dtypes):

    record_forge_op_name("Subtract")

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
