# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.mnist.utils.utils import load_input, load_model
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.xfail(reason="ttir.softmax op requires attribute 'dimension'")
def test_mnist(record_forge_property):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="mnist",
        source=Source.GITHUB,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Record Forge Property
    record_forge_property("group", "generality")
    record_forge_property("tags.model_name", module_name)

    # Load model and input
    framework_model = load_model()
    inputs = load_input()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
