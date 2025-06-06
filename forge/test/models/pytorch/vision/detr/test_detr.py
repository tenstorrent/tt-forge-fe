# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Detr model having both object detection and segmentation model
# https://huggingface.co/docs/transformers/en/model_doc/detr

import pytest
import requests
import torch
from PIL import Image
from transformers import (
    DetrFeatureExtractor,
    DetrForObjectDetection,
    DetrForSegmentation,
    DetrImageProcessor,
)

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify


class DetrWrapper(torch.nn.Module):
    def __init__(self, model, task="detection"):
        super().__init__()
        self.model = model
        assert task in ["detection", "segmentation"], "Task must be 'detection' or 'segmentation'"
        self.task = task

    def forward(self, pixel_values, pixel_mask):
        output = self.model(pixel_values, pixel_mask)
        if self.task == "detection":
            return (output.logits, output.pred_boxes)
        else:
            return (output.logits, output.pred_masks, output.pred_boxes)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("facebook/detr-resnet-50", marks=[pytest.mark.xfail]),
    ],
)
def test_detr_detection(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DETR,
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load the model
    model = DetrForObjectDetection.from_pretrained(variant).to(torch.bfloat16)
    framework_model = DetrWrapper(model, task="detection")

    # Preprocess the image for the model
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    input = processor(images=image, return_tensors="pt")
    inputs = [input["pixel_values"].to(torch.bfloat16), input["pixel_mask"].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.97)),
    )


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "facebook/detr-resnet-50-panoptic",
            marks=[pytest.mark.xfail],
        )
    ],
)
def test_detr_segmentation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DETR,
        variant=variant,
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Load the model
    framework_model = DetrForSegmentation.from_pretrained(variant).to(torch.bfloat16)
    framework_model = DetrWrapper(framework_model, task="segmentation")

    # Preprocess the image for the model
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
    input = feature_extractor(images=image, return_tensors="pt")
    inputs = [input["pixel_values"].to(torch.bfloat16), input["pixel_mask"].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)
