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
from forge.forge_property_utils import Framework, Source, Task, ModelArch, record_model_properties
from test.models.models_utils import preprocess_input_data


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50"])
def test_detr_detection_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DETR,
        variant=variant,
        task=Task.CV_OBJECT_DET,
        source=Source.HUGGINGFACE,
    )

    # Load the model
    framework_model = download_model(DetrForObjectDetection.from_pretrained, variant, return_dict=False)
    framework_model.eval()

    # Prepare input
    input_batch = preprocess_input_data()
    inputs = [input_batch]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/detr_obj_det.onnx"
    torch.onnx.export(framework_model, (inputs[0]), onnx_path, opset_version=17)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50-panoptic"])
def test_detr_segmentation_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DETR,
        variant=variant,
        task=Task.CV_IMAGE_SEG,
        source=Source.HUGGINGFACE,
    )

    # Load the model
    framework_model = download_model(DetrForSegmentation.from_pretrained, variant, return_dict=False)
    framework_model.eval()

    # Prepare input
    input_batch = preprocess_input_data()
    inputs = [input_batch]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/detr_semseg.onnx"
    torch.onnx.export(framework_model, (inputs[0]), onnx_path, opset_version=17)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    _, co_out = verify(inputs, framework_model, compiled_model)
