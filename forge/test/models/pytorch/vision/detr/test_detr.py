# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Detr model having both object detection and segmentation model
# https://huggingface.co/docs/transformers/en/model_doc/detr

import pytest
from transformers import DetrForObjectDetection, DetrForSegmentation

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.detr.utils.image_utils import preprocess_input_data
from test.models.utils import Framework, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50"])
def test_detr_detection(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="detr", variant=variant, task=Task.OBJECT_DETECTION
    )

    record_forge_property("module_name", module_name)

    # Load the model
    framework_model = DetrForObjectDetection.from_pretrained(variant)

    # Preprocess the image for the model
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    input_batch = preprocess_input_data(image_url)

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50-panoptic"])
def test_detr_segmentation(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="detr", variant=variant, task=Task.SEMANTIC_SEGMENTATION
    )

    record_forge_property("module_name", module_name)

    # Load the model
    framework_model = DetrForSegmentation.from_pretrained(variant)

    # Preprocess the image for the model
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    input_batch = preprocess_input_data(image_url)

    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
