# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import onnx
from loguru import logger
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from transformers import (
    AutoImageProcessor,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from test.models.models_utils import print_cls_results


def get_sample_data(model_name):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(input_image))
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    except:
        logger.warning(
            "Failed to download the image file, replacing input with random tensor. Please check if the URL is up to date"
        )
        height = image_processor.to_dict()["size"]["height"]
        width = image_processor.to_dict()["size"]["width"]
        pixel_values = torch.rand(1, 3, height, width).to(torch.float32)
    return pixel_values


variants = [
    pytest.param("deepmind/vision-perceiver-conv", id="deepmind/vision-perceiver-conv", marks=[pytest.mark.test_duration_check]),
    pytest.param(
        "deepmind/vision-perceiver-learned",
        marks=pytest.mark.xfail,
        id="deepmind/vision-perceiver-learned",
    ),
    pytest.param(
        "deepmind/vision-perceiver-fourier",
        marks=pytest.mark.xfail,
        id="deepmind/vision-perceiver-fourier",
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_perceiverio_for_image_classification_onnx(variant, forge_tmp_path):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.ONNX,
        model=ModelArch.PERCEIVERIO,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Sample Image
    pixel_values = get_sample_data(variant)

    # Load the model from HuggingFace
    if variant == "deepmind/vision-perceiver-learned":
        framework_model = PerceiverForImageClassificationLearned.from_pretrained(variant, return_dict=False)

    elif variant == "deepmind/vision-perceiver-conv":
        framework_model = PerceiverForImageClassificationConvProcessing.from_pretrained(variant, return_dict=False)

    elif variant == "deepmind/vision-perceiver-fourier":
        framework_model = PerceiverForImageClassificationFourier.from_pretrained(variant, return_dict=False)

    else:
        logger.info(f"The model {variant} is not supported")

    framework_model.eval()
    inputs = [pixel_values]

    # Export model to ONNX
    onnx_path = f"{forge_tmp_path}/{variant.replace('/', '_')}_perceiver.onnx"
    torch.onnx.export(
        framework_model,
        pixel_values,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    # Load ONNX model and check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Forge compile ONNX model
    compiled_model = forge.compile(
        onnx_model,
        sample_inputs=inputs,
        module_name=module_name,
    )

    # Model Verification
    fw_out, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
    )

    # Post processing
    print_cls_results(fw_out[0], co_out[0])
