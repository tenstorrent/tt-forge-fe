# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from transformers import SegformerForImageClassification, SegformerForSemanticSegmentation, SegformerConfig

from test.models.pytorch.vision.segformer.utils.image_utils import get_sample_data

import forge
from test.models.utils import build_module_name


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
def test_segformer_image_classification_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="segformer", variant=variant, task="imgcls")

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

    compiled_model = forge.compile(model, sample_inputs=[pixel_values], module_name=module_name)


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
def test_segformer_semantic_segmentation_pytorch(record_forge_property, variant):
    module_name = build_module_name(framework="pt", model="segformer", variant=variant, task="semseg")

    # Load the model from HuggingFace
    model = SegformerForSemanticSegmentation.from_pretrained(variant)
    model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)
    compiled_model = forge.compile(model, sample_inputs=[pixel_values], module_name=module_name)
