# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.utils.utils import load_vision_model_and_input
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

dataset = load_dataset("huggingface/cats-image")
image_1 = dataset["test"]["image"][0]


def generate_model_vit_imgcls_hf_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    image_processor = download_model(AutoImageProcessor.from_pretrained, variant)
    model = download_model(ViTForImageClassification.from_pretrained, variant)
    # STEP 3: Run inference on Tenstorrent device
    img_tensor = image_processor(image_1, return_tensors="pt").pixel_values
    # output = model(img_tensor).logits

    return model, [img_tensor], {}


variants = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_vit_classify_224_hf_pytorch(record_forge_property, variant):
    if variant != "google/vit-base-patch16-224":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="vit",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    if variant in ["google/vit-base-patch16-224"]:
        record_forge_property("group", "priority")
    else:
        record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    framework_model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants_with_weights = {
    "vit_b_16": "ViT_B_16_Weights",
    "vit_b_32": "ViT_B_32_Weights",
    "vit_l_16": "ViT_L_16_Weights",
    "vit_l_32": "ViT_L_32_Weights",
    "vit_h_14": "ViT_H_14_Weights",
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_vit_torchvision(record_forge_property, variant):

    if variant != "vit_b_16":
        pytest.skip("Skipping this variant; only testing the small variant(vit_b_16) for now.")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="vit",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "classification", weight_name)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
