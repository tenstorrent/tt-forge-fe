# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

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
from test.models.pytorch.multimodal.clip.model_utils.clip_model import CLIPTextWrapper
from test.utils import download_model


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "openai/clip-vit-base-patch32",
        ),
    ],
)
def test_clip_pytorch(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.CLIP,
        variant=variant,
        suffix="text",
        source=Source.HUGGINGFACE,
        task=Task.TEXT_GENERATION,
    )

    # Load processor and model from HuggingFace
    model = download_model(CLIPModel.from_pretrained, variant, torchscript=True)
    processor = download_model(CLIPProcessor.from_pretrained, variant)

    # Load image from the IAM dataset
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    image = next(iter(dataset.skip(10)))["image"]

    # Process image
    text = [
        "a photo of a cat",
        "a photo of a dog",
    ]
    inputs = processor(text=text, images=image, return_tensors="pt")

    inputs = [inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]]
    framework_model = CLIPTextWrapper(model)
    inputs = [inputs[0], inputs[2]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model)

    # Model Postprocessing
    print_cls_results(fw_out[0], co_out[0])
