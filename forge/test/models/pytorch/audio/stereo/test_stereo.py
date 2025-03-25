# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.verify.verify import verify

from .utils import load_inputs, load_model
from test.models.utils import Framework, Source, Task, build_module_name

variants = [
    pytest.param(
        "facebook/musicgen-small",
        marks=[
            pytest.mark.xfail(
                reason="[Optimization Graph Passes] RuntimeError: (i >= 0) && (i < (int)dims_.size()) Trying to access element outside of dimensions: 3"
            )
        ],
    ),
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_stereo(forge_property_recorder, variant):
    if variant != "facebook/musicgen-small":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="stereo",
        variant=variant,
        task=Task.MUSIC_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

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
