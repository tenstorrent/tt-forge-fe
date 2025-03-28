# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from gliner import GLiNER

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["urchade/gliner_multi-v2.1"]


@pytest.mark.nightly
@pytest.mark.xfail(reason="IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)")
@pytest.mark.parametrize("variant", variants)
def test_gliner(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="Gliner",
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_model_name(module_name)

    # Load model
    framework_model = GLiNER.from_pretrained(variant)
    framework_model.eval()

    text = """
    Cristiano Ronaldo dos Santos Aveiro was born 5 February 1985) is a Portuguese professional footballer.
    """
    labels = ["person", "award", "date", "competitions", "teams"]

    text_encoded = torch.tensor([ord(c) for c in text], dtype=torch.int64).unsqueeze(0)
    label_tensor = torch.tensor(list(range(len(labels))), dtype=torch.int64)

    # prepare input
    inputs = [text_encoded, label_tensor]

    # prepare input
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
