# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from .utils import load_inputs, load_model

variants = [
    pytest.param(
        "facebook/musicgen-small",
    ),
    pytest.param(
        "facebook/musicgen-medium",
    ),
    pytest.param(
        "facebook/musicgen-large",
        marks=pytest.mark.skip(
            reason="Insufficient host DRAM to run this model (requires a bit more than 26 GB during compile time)"
        ),
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_stereo(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="stereo",
        variant=variant,
        task=Task.MUSIC_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    framework_model, processor = load_model(variant)

    input_ids, attn_mask, decoder_input_ids = load_inputs(framework_model, processor)
    inputs = [input_ids, attn_mask, decoder_input_ids]

    # Issue: https://github.com/tenstorrent/tt-forge-fe/issues/615
    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
