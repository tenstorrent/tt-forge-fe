# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from datasets import load_dataset
from transformers import AutoFeatureExtractor, ViTForImageClassification

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from test.utils import download_model


def generate_model_deit_imgcls_hf_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    image_processor = download_model(AutoFeatureExtractor.from_pretrained, variant)
    model = download_model(ViTForImageClassification.from_pretrained, variant, return_dict=False)

    # STEP 3: Run inference on Tenstorrent device
    dataset = load_dataset("huggingface/cats-image")
    image_1 = dataset["test"]["image"][0]
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
@pytest.mark.parametrize("variant", variants)
def test_deit_imgcls_hf_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="deit",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    framework_model, inputs, _ = generate_model_deit_imgcls_hf_pytorch(
        variant,
    )
    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    pcc = 0.99
    if variant == "facebook/deit-base-patch16-224":
        pcc = 0.96

    # Model Verification and inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post processing
    logits = co_out[0]
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", framework_model.config.id2label[predicted_class_idx])
