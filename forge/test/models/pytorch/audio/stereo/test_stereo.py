# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.verify.verify import verify

from .utils import load_inputs, load_model
from test.models.utils import Framework, build_module_name

variants = [
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_stereo(record_forge_property, variant):
    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="stereo", variant=variant)

    # Record Forge Property
    record_forge_property("module_name", module_name)

    framework_model, processor = load_model(variant)

    input_ids, attn_mask, decoder_input_ids = load_inputs(framework_model, processor)
    inputs = [input_ids, attn_mask, decoder_input_ids]

    # Issue: https://github.com/tenstorrent/tt-forge-fe/issues/615
    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
