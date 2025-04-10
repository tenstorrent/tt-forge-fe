# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Detr model having both object detection and segmentation model
# https://huggingface.co/docs/transformers/en/model_doc/detr

import pytest
from transformers import DetrForObjectDetection, DetrForSegmentation

import forge
import torch
import onnx
from forge.verify.verify import verify
from test.utils import download_model
from forge.forge_property_utils import Framework, Source, Task
from test.models.models_utils import preprocess_input_data


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50"])
def test_detr_detection_onnx(forge_property_recorder, variant, tmp_path):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="detr",
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")

    # Load the model
    framework_model = download_model(DetrForObjectDetection.from_pretrained, variant, return_dict=False)
    framework_model.eval()

    # Prepare input
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    input_batch = preprocess_input_data(image_url)
    inputs = [input_batch]

    # Export model to ONNX
    onnx_path = f"{tmp_path}/detr_obj_det.onnx"
    torch.onnx.export(framework_model, (inputs[0]), onnx_path, opset_version=17)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50-panoptic"])
def test_detr_segmentation_onnx(forge_property_recorder, variant, tmp_path):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="detr",
        variant=variant,
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")

    # Load the model
    framework_model = download_model(DetrForSegmentation.from_pretrained, variant, return_dict=False)
    framework_model.eval()

    # Prepare input
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    input_batch = preprocess_input_data(image_url)
    inputs = [input_batch]

    # Export model to ONNX
    onnx_path = f"{tmp_path}/detr_semseg.onnx"
    torch.onnx.export(framework_model, (inputs[0]), onnx_path, opset_version=17)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
