# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import (
    SegformerConfig,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
)

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.segformer.utils.image_utils import get_sample_data
from test.models.utils import Framework, Source, Task, build_module_name

variants_img_classification = [
    "nvidia/mit-b0",
    "nvidia/mit-b1",
    "nvidia/mit-b2",
    "nvidia/mit-b3",
    "nvidia/mit-b4",
    "nvidia/mit-b5",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_img_classification)
def test_segformer_image_classification_pytorch(record_forge_property, variant):
    if variant != "nvidia/mit-b0":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="segformer",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Set model configurations
    config = SegformerConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = SegformerConfig(**config_dict)

    # Load the model from HuggingFace
    framework_model = SegformerForImageClassification.from_pretrained(variant, config=config)
    framework_model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)
    inputs = [pixel_values]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants_semseg = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_semseg)
def test_segformer_semantic_segmentation_pytorch(record_forge_property, variant):
    pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="segformer",
        variant=variant,
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load the model from HuggingFace
    framework_model = SegformerForSemanticSegmentation.from_pretrained(variant)
    framework_model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)
    inputs = [pixel_values]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
