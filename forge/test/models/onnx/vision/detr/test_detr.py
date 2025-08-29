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
from third_party.tt_forge_models.tools.utils import get_file
from PIL import Image
import numpy as np
from torchvision import transforms


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50"])
def test_detr_detection_onnx(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DETR,
        variant=variant,
        task=Task.OBJECT_DETECTION,
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
        task=Task.SEMANTIC_SEGMENTATION,
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


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50"])
def test_detr_onnx_torchhub(variant, forge_tmp_path):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.DETR,
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.TORCH_HUB,
    )

    # Load the model
    torch_model = torch.hub.load("facebookresearch/detr:main", "detr_resnet50", pretrained=True)
    torch_model.eval()

    # Prepare input
    image_file = get_file(
        "https://huggingface.co/spaces/nakamura196/yolov5-char/resolve/8a166e0aa4c9f62a364dafa7df63f2a33cbb3069/ultralytics/yolov5/data/images/zidane.jpg"
    )
    input_image = Image.open(str(image_file))
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    inputs = [input_batch]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/detr_obj_det_torch_hub.onnx"
    torch.onnx.export(torch_model, (inputs[0],), onnx_path, opset_version=17)

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(onnx_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
