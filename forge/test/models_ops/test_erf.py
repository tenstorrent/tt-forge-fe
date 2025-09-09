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


class Erf0(ForgeModule):
    def __init__(self, name):
        super().__init__(name)

    def forward(self, erf_input_0):
        erf_output_1 = forge.op.Erf("", erf_input_0)
        return erf_output_1


def ids_func(param):
    forge_module = param[0]
    shapes_dtypes = param[1]
    return str(forge_module.__name__) + "-" + str(shapes_dtypes)


forge_modules_and_shapes_dtypes_list = [
    (Erf0, [((1, 128, 16384), torch.float32)], {"model_names": ["onnx_albert_xxlarge_v1_mlm_hf"], "pcc": 0.99}),
    (
        Erf0,
        [((1, 128, 128), torch.float32)],
        {
            "model_names": [
                "onnx_albert_xxlarge_v1_mlm_hf",
                "onnx_albert_large_v1_mlm_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_albert_xlarge_v1_mlm_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 512, 256), torch.float32)],
        {
            "model_names": ["onnx_mlp_mixer_mixer_s16_224_img_cls_timm", "onnx_mlp_mixer_mixer_s32_224_img_cls_timm"],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 196, 2048), torch.float32)],
        {"model_names": ["onnx_mlp_mixer_mixer_s16_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 16384, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 4096, 512), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 1024, 1280), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 256, 2048), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b2_img_cls_hf",
                "onnx_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 13, 1536), torch.float32)],
        {"model_names": ["onnx_minilm_sentence_transformers_all_minilm_l6_v2_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 768, 384), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_b16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b32_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 196, 3072), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_b16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_in21k_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_miil_img_cls_timm",
                "onnx_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 384, 3000), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 384, 1500), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1500, 1536), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1, 1536), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_tiny_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 128, 3072), torch.float32)],
        {
            "model_names": [
                "onnx_bert_bert_base_uncased_mlm_hf",
                "onnx_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf",
                "onnx_albert_base_v1_mlm_hf",
                "onnx_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (Erf0, [((1, 128, 768), torch.float32)], {"model_names": ["onnx_bert_bert_base_uncased_mlm_hf"], "pcc": 0.99}),
    (
        Erf0,
        [((1, 257, 3072), torch.float32)],
        {"model_names": ["onnx_mgp_alibaba_damo_mgp_str_base_scene_text_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1024, 512), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_l32_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_l16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 49, 4096), torch.float32)],
        {"model_names": ["onnx_mlp_mixer_mixer_l32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 256, 1280), torch.float32)],
        {"model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 2048, 768), torch.float32)],
        {"model_names": ["onnx_perceiverio_deepmind_language_perceiver_mlm_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 16384, 128), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 4096, 256), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 1024, 640), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 256, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf",
                "onnx_segformer_nvidia_mit_b0_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 512, 3000), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 512, 1500), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1500, 2048), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1, 2048), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_base_speech_recognition_hf"], "pcc": 0.99},
    ),
    (Erf0, [((1, 128, 4096), torch.float32)], {"model_names": ["onnx_albert_large_v1_mlm_hf"], "pcc": 0.99}),
    (
        Erf0,
        [((1, 49, 3072), torch.float32)],
        {"model_names": ["onnx_mlp_mixer_mixer_b32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 512, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 1, 1024), torch.float32)],
        {
            "model_names": [
                "onnx_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf",
                "onnx_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 3072, 128), torch.float32)],
        {"model_names": ["onnx_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 201, 3072), torch.float32)],
        {"model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1536), torch.float32)],
        {"model_names": ["onnx_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 196, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_mlp_mixer_mixer_l16_224_img_cls_timm",
                "onnx_mlp_mixer_mixer_l16_224_in21k_img_cls_timm",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 4096, 384), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1024, 768), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 256, 1536), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 64, 3072), torch.float32)],
        {"model_names": ["onnx_swin_microsoft_swinv2_tiny_patch4_window8_256_masked_img_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 197, 3072), torch.float32)],
        {
            "model_names": [
                "onnx_deit_facebook_deit_base_patch16_224_img_cls_hf",
                "onnx_beit_microsoft_beit_base_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_base_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 197, 1536), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_small_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 197, 4096), torch.float32)],
        {
            "model_names": [
                "onnx_beit_microsoft_beit_large_patch16_224_img_cls_hf",
                "onnx_vit_base_google_vit_large_patch16_224_img_cls_hf",
            ],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 197, 768), torch.float32)],
        {"model_names": ["onnx_deit_facebook_deit_tiny_patch16_224_img_cls_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 49, 2048), torch.float32)],
        {"model_names": ["onnx_mlp_mixer_mixer_s32_224_img_cls_timm"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 768, 3000), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 768, 1500), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1500, 3072), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1, 3072), torch.float32)],
        {"model_names": ["onnx_whisper_openai_whisper_small_speech_recognition_hf"], "pcc": 0.99},
    ),
    (Erf0, [((1, 128, 8192), torch.float32)], {"model_names": ["onnx_albert_xlarge_v1_mlm_hf"], "pcc": 0.99}),
    (
        Erf0,
        [((1, 6, 3072), torch.float32)],
        {
            "model_names": ["onnx_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf"],
            "pcc": 0.99,
        },
    ),
    (
        Erf0,
        [((1, 19200, 256), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 4800, 512), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 1200, 1280), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 300, 2048), torch.float32)],
        {"model_names": ["onnx_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf"], "pcc": 0.99},
    ),
    (
        Erf0,
        [((1, 384, 4096), torch.float32)],
        {"model_names": ["onnx_bert_phiyodr_bert_large_finetuned_squad2_qa_hf"], "pcc": 0.99},
    ),
]


@pytest.mark.nightly_models_ops
@pytest.mark.parametrize("forge_module_and_shapes_dtypes", forge_modules_and_shapes_dtypes_list, ids=ids_func)
@pytest.mark.parametrize("training_test", [False, True], ids=["inference", "training"])
def test_module(forge_module_and_shapes_dtypes, training_test):

    record_forge_op_name("Erf")

    forge_module, operand_shapes_dtypes, metadata = forge_module_and_shapes_dtypes

    pcc = metadata.get("pcc")

    for metadata_name, metadata_value in metadata.items():
        if metadata_name in ["pcc"]:
            continue
        elif metadata_name == "model_names":
            record_op_model_names(metadata_value)
        elif metadata_name == "args":
            record_forge_op_args(metadata_value)
        else:
            logger.warning(
                "No utility function available in forge property handler to record %s property", metadata_name
            )

    max_int = 1000
    inputs = [
        Tensor.create_from_shape(operand_shape, operand_dtype, max_int=max_int, requires_grad=training_test)
        for operand_shape, operand_dtype in operand_shapes_dtypes
    ]

    framework_model = forge_module(forge_module.__name__)

    for name, parameter in framework_model._parameters.items():
        parameter_tensor = Tensor.create_torch_tensor(
            shape=parameter.shape.get_pytorch_shape(),
            dtype=parameter.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_parameter(name, parameter_tensor)

    for name, constant in framework_model._constants.items():
        constant_tensor = Tensor.create_torch_tensor(
            shape=constant.shape.get_pytorch_shape(),
            dtype=constant.pt_data_format,
            max_int=max_int,
            requires_grad=training_test,
        )
        framework_model.set_constant(name, constant_tensor)

    record_single_op_operands_info(framework_model, inputs)

    compiler_cfg = forge.config.CompilerConfig()
    if "default_df_override" in metadata.keys():
        compiler_cfg.default_df_override = forge.DataFormat.from_json(metadata["default_df_override"])

    compiled_model = compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg, training=training_test)

    verify(
        inputs,
        framework_model,
        compiled_model,
        with_backward=training_test,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )
