# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model

dataset = load_dataset("huggingface/cats-image")
image_1 = dataset["test"]["image"][0]
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_2 = Image.open(requests.get(url, stream=True).raw)


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
    record_forge_property("model_name", module_name)

    framework_model, inputs, _ = generate_model_vit_imgcls_hf_pytorch(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
