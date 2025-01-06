# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import requests
from PIL import Image
from datasets import load_dataset

from transformers import AutoFeatureExtractor, ViTForImageClassification

import forge
from test.utils import download_model
from test.models.utils import build_module_name, Framework, Task
from forge.verify.verify import verify


def generate_model_deit_imgcls_hf_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    image_processor = download_model(AutoFeatureExtractor.from_pretrained, variant)
    model = download_model(ViTForImageClassification.from_pretrained, variant)

    # STEP 3: Run inference on Tenstorrent device
    dataset = load_dataset("huggingface/cats-image")
    image_1 = dataset["test"]["image"][0]
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_2 = Image.open(requests.get(url, stream=True).raw)
    img_tensor = image_processor(image_1, return_tensors="pt").pixel_values
    # output = model(img_tensor).logits

    return model, [img_tensor], {}


variants = [
    "facebook/deit-base-patch16-224",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/deit-tiny-patch16-224",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_deit_imgcls_hf_pytorch(record_forge_property, variant):
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="deit", variant=variant, task=Task.IMAGE_CLASSIFICATION
    )

    record_forge_property("module_name", module_name)

    framework_model, inputs, _ = generate_model_deit_imgcls_hf_pytorch(
        variant,
    )
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    verify(inputs, framework_model, compiled_model)
