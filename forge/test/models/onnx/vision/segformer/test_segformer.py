# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge
from transformers import AutoImageProcessor
import os
import pytest
import onnx
import torch
from forge.verify.verify import verify
from test.models.utils import Framework, Source, Task, build_module_name
from transformers import SegformerForSemanticSegmentation, SegformerForImageClassification
from test.models.models_utils import get_sample_data
from test.utils import download_model

variants_img_classification = [
    pytest.param(
        "nvidia/mit-b0",
    ),
    "nvidia/mit-b1",
    "nvidia/mit-b2",
    "nvidia/mit-b3",
    "nvidia/mit-b4",
    "nvidia/mit-b5",
]


@pytest.mark.parametrize("variant", variants_img_classification)
@pytest.mark.nightly
def test_segformer_image_classification_onnx(forge_property_recorder, variant, tmp_path):
    if variant != "nvidia/mit-b0":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="segformer",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    if variant == "nvidia/mit-b0":
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")

    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

    # Load the model from HuggingFace
    torch_model = download_model(SegformerForImageClassification.from_pretrained, variant, return_dict=False)
    torch_model.eval()

    # prepare input
    inputs = get_sample_data(variant)

    # Export model to ONNX
    onnx_path = f"{tmp_path}/segformer_" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(
        onnx_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )


variants_semseg = [
    pytest.param(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        marks=[pytest.mark.xfail],
    ),
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
]


@pytest.mark.parametrize("variant", variants_semseg)
@pytest.mark.nightly
def test_segformer_semantic_segmentation_onnx(forge_property_recorder, variant, tmp_path):
    if variant != "nvidia/segformer-b0-finetuned-ade-512-512":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.ONNX,
        model="segformer",
        variant=variant,
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    if variant == "nvidia/segformer-b0-finetuned-ade-512-512":
        forge_property_recorder.record_group("red")
    else:
        forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # Load the model from HuggingFace
    torch_model = download_model(SegformerForSemanticSegmentation.from_pretrained, variant, return_dict=False)
    torch_model.eval()

    # prepare input
    inputs = get_sample_data(variant)

    # Export model to ONNX
    onnx_path = f"{tmp_path}/" + str(variant).split("/")[-1].replace("-", "_") + ".onnx"
    torch.onnx.export(torch_model, inputs[0], onnx_path, opset_version=17)

    # Load framework model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Compile model
    compiled_model = forge.compile(
        onnx_model, inputs, forge_property_handler=forge_property_recorder, module_name=module_name
    )

    # Model Verification
    verify(
        inputs,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
