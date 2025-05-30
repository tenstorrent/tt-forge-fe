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
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=3)
        return softmax_output_1


class Softmax1(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, softmax_input_0):
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=-1)
        return softmax_output_1


class Softmax2(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, softmax_input_0):
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=2)
        return softmax_output_1


class Softmax3(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, softmax_input_0):
        softmax_output_1 = forge.op.Softmax("", softmax_input_0, dim=1)
        return softmax_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (
        Softmax0,
        [((1, 12, 6, 6), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax1,
        [((1, 12, 6, 6), torch.float32)],
        {
            "model_names": ["pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax2,
        [((8, 100, 100), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Softmax2,
        [((8, 280, 280), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Softmax2,
        [((8, 100, 280), torch.float32)],
        {
            "model_names": [
                "onnx_detr_facebook_detr_resnet_50_obj_det_hf",
                "onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "2"},
        },
    ),
    (
        Softmax0,
        [((1, 1, 16384, 256), torch.float32)],
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
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax0,
        [((1, 2, 4096, 256), torch.float32)],
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
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax0,
        [((1, 5, 1024, 256), torch.float32)],
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
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax0,
        [((1, 8, 256, 256), torch.float32)],
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
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax1,
        [((1, 12, 128, 128), torch.float32)],
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
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax0,
        [((1, 12, 128, 128), torch.float32)],
        {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Softmax1,
        [((1, 16, 128, 128), torch.float32)],
        {
            "model_names": [
                "pt_albert_xlarge_v2_token_cls_hf",
                "pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf",
                "pt_albert_large_v1_mlm_hf",
                "pt_albert_large_v2_token_cls_hf",
                "pt_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax1,
        [((1, 64, 128, 128), torch.float32)],
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
        Softmax1,
        [((16, 256, 256), torch.float32)],
        {"model_names": ["pt_bart_facebook_bart_large_mnli_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 16, 256, 256), torch.float32)],
        {
            "model_names": [
                "pt_codegen_salesforce_codegen_350m_multi_clm_hf",
                "pt_codegen_salesforce_codegen_350m_nl_clm_hf",
                "pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax1,
        [((1, 16, 32, 32), torch.float32)],
        {
            "model_names": ["pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf", "pt_bloom_bigscience_bloom_1b1_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax1,
        [((1, 32, 256, 256), torch.float32)],
        {
            "model_names": ["pt_llama3_meta_llama_llama_3_2_1b_clm_hf", "pt_phi2_microsoft_phi_2_clm_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax1,
        [((12, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_125m_qa_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 12, 204, 204), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_mlm_mlm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 12, 1, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 12, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 12, 1, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 16, 384, 384), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Softmax1,
        [((1, 16, 384, 384), torch.float32)],
        {"model_names": ["pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 12, 13, 13), torch.float32)],
        {
            "model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax1,
        [((16, 7, 7), torch.float32)],
        {"model_names": ["pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 12, 384, 384), torch.float32)],
        {
            "model_names": ["pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf"],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax1,
        [((1, 12, 7, 7), torch.float32)],
        {
            "model_names": [
                "pt_gpt_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf",
                "pt_nanogpt_financialsupport_nanogpt_text_gen_hf",
            ],
            "pcc": 0.99,
            "args": {"dim": "-1"},
        },
    ),
    (
        Softmax1,
        [((1, 32, 4, 4), torch.float32)],
        {"model_names": ["pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((32, 256, 256), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_1_3b_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((16, 32, 32), torch.float32)],
        {"model_names": ["pt_opt_facebook_opt_350m_qa_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 32, 7, 7), torch.float32)],
        {"model_names": ["pt_phi_1_5_microsoft_phi_1_5_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 12, 201, 201), torch.float32)],
        {"model_names": ["pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 6, 1, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 6, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 6, 1, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax2,
        [((1, 100, 2240), torch.float32)],
        {"model_names": ["onnx_detr_facebook_detr_resnet_50_panoptic_sem_seg_hf"], "pcc": 0.99, "args": {"dim": "2"}},
    ),
    (
        Softmax0,
        [((1, 12, 197, 197), torch.float32)],
        {"model_names": ["onnx_vit_base_google_vit_base_patch16_224_img_cls_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Softmax3,
        [((1, 16, 4, 8400), torch.float32)],
        {"model_names": ["onnx_yolov8_default_obj_det_github"], "pcc": 0.99, "args": {"dim": "1"}},
    ),
    (
        Softmax1,
        [((1, 20, 2, 2), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 20, 2, 2), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Softmax1,
        [((1, 20, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 20, 1500, 1500), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Softmax1,
        [((1, 20, 2, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax0,
        [((1, 20, 2, 1500), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_large_v3_clm_hf"], "pcc": 0.99, "args": {"dim": "3"}},
    ),
    (
        Softmax0,
        [((64, 3, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax0,
        [((16, 6, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax0,
        [((4, 12, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax0,
        [((1, 24, 64, 64), torch.float32)],
        {
            "model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_img_cls_hf"],
            "pcc": 0.99,
            "args": {"dim": "3"},
        },
    ),
    (
        Softmax1,
        [((1, 12, 14, 14), torch.float32)],
        {"model_names": ["pt_albert_twmkn9_albert_base_v2_squad2_qa_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 16, 588, 588), torch.float32)],
        {"model_names": ["pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((48, 1, 1), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((2, 12, 13, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((48, 1, 13), torch.float32)],
        {"model_names": ["pt_stereo_facebook_musicgen_medium_music_generation_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 8, 1, 1), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 8, 1500, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
    ),
    (
        Softmax1,
        [((1, 8, 1, 1500), torch.float32)],
        {"model_names": ["pt_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99, "args": {"dim": "-1"}},
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
