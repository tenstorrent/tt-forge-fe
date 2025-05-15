# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from datasets import load_dataset
from transformers import AutoFeatureExtractor, ViTForImageClassification, DeiTConfig

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.utils import download_model


def generate_model_deit_imgcls_hf_pytorch(variant):
    # STEP 2: Create Forge module from PyTorch model
    image_processor = download_model(AutoFeatureExtractor.from_pretrained, variant)
    model = download_model(ViTForImageClassification.from_pretrained, variant)

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
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_deit_imgcls_hf_pytorch(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="deit",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    framework_model, inputs, _ = generate_model_deit_imgcls_hf_pytorch(
        variant,
    )
    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)


def test_customer_team_tiny_deit(forge_property_recorder):
    sequence_size = 224
    image_channels = 3
    image_size = 224

    model_name = "facebook/deit-tiny-patch16-224"
    config = DeiTConfig.from_pretrained(model_name)
    batch_size = 1
    config.num_hidden_layers = 12
    model = ViTForImageClassification.from_pretrained(model_name, config=config)
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = AutoFeatureExtractor.from_pretrained("facebook/deit-tiny-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)
    compiled_model = forge.compile(model, sample_inputs=[torch_pixel_values])

    # file_path = "generated_export_deit_tiny.cpp"
    # compiled_model.export_to_cpp(file_path)

    # Inference with compiled model
    output = compiled_model(torch_pixel_values)