# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.verify.verify import verify

from .utils import load_inputs, load_model
from test.models.utils import Framework, Source, Task, build_module_name

variants = ["llava-hf/llava-1.5-7b-hf"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_llava(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="llava",
        variant=variant,
        task=Task.CONDITIONAL_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    framework_model, processor = load_model(variant)
    image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    text = "What’s shown in this image?"

    # Input sample
    input_ids, attn_mask, pixel_values = load_inputs(image, text, processor)
    inputs = [input_ids, attn_mask, pixel_values]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
