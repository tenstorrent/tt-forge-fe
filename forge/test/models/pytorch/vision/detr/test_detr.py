# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Detr model having both object detection and segmentation model
# https://huggingface.co/docs/transformers/en/model_doc/detr

import pytest
import torch
from transformers import DetrForObjectDetection, DetrForSegmentation

import forge
from forge.forge_property_utils import Framework, Source, Task, ModelGroup, ModelPriority
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from test.models.pytorch.vision.detr.utils.image_utils import preprocess_input_data


class DetrWrapper(torch.nn.Module):
    def __init__(self, model, task="detection"):
        super().__init__()
        self.model = model
        assert task in ["detection", "segmentation"], "Task must be 'detection' or 'segmentation'"
        self.task = task

    def forward(self, input_batch):
        output = self.model(input_batch)
        if self.task == "detection":
            return output.logits
        else:
            return (output.logits, output.pred_masks, output.pred_boxes)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    ["facebook/detr-resnet-50"],
)
def test_detr_detection(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="detr",
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load the model
    framework_model = DetrForObjectDetection.from_pretrained(variant)
    framework_model = DetrWrapper(framework_model, task="detection")

    # Preprocess the image for the model
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    input_batch = preprocess_input_data(image_url)

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        forge_property_handler=forge_property_recorder,
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
def test_detr_segmentation(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="detr",
        variant=variant,
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Load the model
    framework_model = DetrForSegmentation.from_pretrained(variant)
    framework_model = DetrWrapper(framework_model, task="segmentation")

    # Preprocess the image for the model
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    input_batch = preprocess_input_data(image_url)

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
