# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.vision.mobilenet.utils.mobilenet_v3_ssd_utils import (
    load_input,
    load_model,
)

variants_with_weights = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_mobilenetv3_ssd(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mobilenetv3_ssd",
        variant=variant,
        source=Source.TORCHVISION,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model = load_model(variant, weight_name)
    inputs = load_input()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
