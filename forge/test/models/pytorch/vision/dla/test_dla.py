# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.pytorch.vision.dla.model_utils.utils import load_dla_model
from test.models.pytorch.vision.vision_utils.utils import load_timm_model_and_input

variants = [
    pytest.param("dla34"),
    pytest.param("dla46_c"),
    pytest.param("dla46x_c"),
    pytest.param("dla60"),
    pytest.param("dla60x"),
    pytest.param("dla60x_c"),
    pytest.param("dla102", marks=[pytest.mark.xfail]),
    pytest.param("dla102x"),
    pytest.param("dla102x2"),
    pytest.param("dla169", marks=[pytest.mark.xfail]),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_dla_pytorch(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model="dla", variant=variant, task=Task.VISUAL_BACKBONE, source=Source.TORCHVISION
    )

    # Load the model and prepare input data
    framework_model, inputs = load_dla_model(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = ["dla34.in1k"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_dla_timm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="dla",
        variant=variant,
        source=Source.TIMM,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load the model and inputs
    framework_model, inputs = load_timm_model_and_input(variant)

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
