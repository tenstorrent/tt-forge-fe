# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.utils.utils import load_vision_model_and_input
from test.models.utils import Framework, Source, Task, build_module_name

variants_with_weights = {
    "ssd300_vgg16": "SSD300_VGG16_Weights",
}


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_ssd300_vgg16(record_forge_property, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="ssd300_vgg16",
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Record Forge Property
    record_forge_property("tags.model_name", module_name)

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "detection", weight_name)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
