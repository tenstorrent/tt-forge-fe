# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from transformers import SegformerForImageClassification, SegformerForSemanticSegmentation, SegformerConfig

from test.models.pytorch.vision.segformer.utils.image_utils import get_sample_data

import forge


variants_img_classification = [
    "nvidia/mit-b0",
    "nvidia/mit-b1",
    "nvidia/mit-b2",
    "nvidia/mit-b3",
    "nvidia/mit-b4",
    "nvidia/mit-b5",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants_img_classification)
def test_segformer_image_classification_pytorch(test_device, variant):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Set model configurations
    config = SegformerConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = SegformerConfig(**config_dict)

    # Load the model from HuggingFace
    model = SegformerForImageClassification.from_pretrained(variant, config=config)
    model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)

    compiled_model = forge.compile(
        model, sample_inputs=[pixel_values], module_name="pt_" + str(variant.split("/")[-1].replace("-", "_")), compiler_cfg=compiler_cfg
    )


variants_semseg = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants_semseg)
def test_segformer_semantic_segmentation_pytorch(test_device, variant):

    # Set Forge configuration parameters
    compiler_cfg = forge.config._get_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # Load the model from HuggingFace
    model = SegformerForSemanticSegmentation.from_pretrained(variant)
    model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)
    compiled_model = forge.compile(
        model, sample_inputs=[pixel_values], module_name="pt_" + str(variant.split("/")[-1].replace("-", "_")), compiler_cfg=compiler_cfg
    )
