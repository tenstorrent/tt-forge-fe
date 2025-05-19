# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import verify

from test.models.pytorch.audio.stereo.model_utils.utils import load_inputs, load_model

variants = [
    pytest.param(
        "facebook/musicgen-small",
        # marks=pytest.mark.xfail,
    ),
    pytest.param(
        "facebook/musicgen-medium",
        # marks=pytest.mark.xfail,
    ),
    pytest.param(
        "facebook/musicgen-large",
        marks=pytest.mark.skip(
            reason="Insufficient host DRAM to run this model (requires a bit more than 26 GB during compile time)"
        ),
    ),
]


@pytest.mark.nightly
# @pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_stereo(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="stereo",
        variant=variant,
        task=Task.MUSIC_GENERATION,
        source=Source.HUGGINGFACE,
    )

    framework_model, processor = load_model(variant)

    input_ids, attn_mask, decoder_input_ids = load_inputs(framework_model, processor)
    inputs = [input_ids, attn_mask, decoder_input_ids]

    # Issue: https://github.com/tenstorrent/tt-forge-fe/issues/615
    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
