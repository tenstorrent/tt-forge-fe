# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import (
    SegformerConfig,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
)

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.pytorch.vision.segformer.model_utils.image_utils import get_sample_data

variants_img_classification = [
    pytest.param("nvidia/mit-b0", marks=pytest.mark.push),
    pytest.param("nvidia/mit-b1", marks=pytest.mark.xfail),
    pytest.param("nvidia/mit-b2", marks=pytest.mark.xfail),
    pytest.param("nvidia/mit-b3", marks=pytest.mark.xfail),
    pytest.param("nvidia/mit-b4", marks=pytest.mark.xfail),
    pytest.param("nvidia/mit-b5", marks=pytest.mark.xfail),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_img_classification)
def test_segformer_image_classification_pytorch(variant):

    if variant in ["nvidia/mit-b0"]:
        group = ModelGroup.RED
        priority = ModelPriority.P1
    else:
        group = ModelGroup.GENERALITY
        priority = ModelPriority.P2

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="segformer",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
        group=group,
        priority=priority,
    )

    # Set model configurations
    config = SegformerConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = SegformerConfig(**config_dict)

    # Load the model from HuggingFace
    framework_model = SegformerForImageClassification.from_pretrained(variant, config=config).to(torch.bfloat16)
    framework_model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)
    inputs = [pixel_values.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post processing
    logits = co_out[0]
    predicted_label = logits.argmax(-1).item()
    print("Predicted class: ", framework_model.config.id2label[predicted_label])


variants_semseg = [
    pytest.param("nvidia/segformer-b0-finetuned-ade-512-512", marks=pytest.mark.xfail),
    pytest.param("nvidia/segformer-b1-finetuned-ade-512-512", marks=pytest.mark.xfail),
    pytest.param("nvidia/segformer-b2-finetuned-ade-512-512", marks=pytest.mark.xfail),
    pytest.param("nvidia/segformer-b3-finetuned-ade-512-512", marks=pytest.mark.xfail),
    pytest.param("nvidia/segformer-b4-finetuned-ade-512-512", marks=pytest.mark.xfail),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_semseg)
def test_segformer_semantic_segmentation_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="segformer",
        variant=variant,
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Load the model from HuggingFace
    framework_model = SegformerForSemanticSegmentation.from_pretrained(variant).to(torch.bfloat16)
    framework_model.eval()

    # Load the sample image
    pixel_values = get_sample_data(variant)
    inputs = [pixel_values.to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
