# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import (
    Swinv2ForMaskedImageModeling,
    Swinv2ForImageClassification,
    Swinv2Model,
    ViTImageProcessor,
)
import onnx
import torch
import forge

from test.models.pytorch.vision.swin.utils.image_utils import load_image
from test.models.pytorch.vision.utils.utils import load_vision_model_and_input
from forge.forge_property_utils import Framework, Source, Task


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
@pytest.mark.xfail
def test_swin_v2_tiny_image_classification_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="swin",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P1")

    # Load the model
    framework_model = Swinv2ForImageClassification.from_pretrained(variant)
    framework_model.eval()

    # Prepare input data
    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Export model to ONNX
    onnx_path = f"{tmp_path}/swin_v2_obj_cls.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_4_256_hf_pytorch(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="swin",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    forge_property_recorder.record_group("generality")

    # Load the model
    framework_model = Swinv2Model.from_pretrained(variant)
    framework_model.eval()

    # Prepare input data
    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Export model to ONNX
    onnx_path = f"{tmp_path}/swin_v2_tiny_4_256.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["microsoft/swinv2-tiny-patch4-window8-256"])
def test_swin_v2_tiny_masked_onnx(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="swin",
        variant=variant,
        task=Task.MASKED_IMAGE_MODELLING,
        source=Source.HUGGINGFACE,
    )
    forge_property_recorder.record_group("generality")

    # Load the model
    framework_model = Swinv2ForMaskedImageModeling.from_pretrained(variant)
    framework_model.eval()

    # Prepare input data
    feature_extractor = ViTImageProcessor.from_pretrained(variant)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = load_image(url, feature_extractor)

    # Export model to ONNX
    onnx_path = f"{tmp_path}/swin_v2_tiny_masked.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )


variants_with_weights = {"swin_v2_t": "Swin_V2_T_Weights"}


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", ["swin_v2_t"])
def test_swin_torchvision(forge_property_recorder, variant, tmp_path):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.ONNX,
        model="swin",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )
    forge_property_recorder.record_group("generality")

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)

    # Export model to ONNX
    onnx_path = f"{tmp_path}/swin_v2_torchvision.onnx"
    torch.onnx.export(
        framework_model, inputs[0], onnx_path, opset_version=17, input_names=["input"], output_names=["output"]
    )

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile framework model
    compiled_model = forge.compile(
        onnx_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )
