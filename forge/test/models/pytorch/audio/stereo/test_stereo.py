# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.verify.verify import verify

from .utils import load_inputs, load_model


variants = [
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
@pytest.mark.xfail(reason="[optimized_graph] Trying to access element outside of dimensions: 3")
def test_stereo(variant):
    # Issue: https://github.com/tenstorrent/tt-forge-fe/issues/615

    framework_model, processor = load_model(variant)

    input_ids, attn_mask, decoder_input_ids = load_inputs(framework_model, processor)
    inputs = [input_ids, attn_mask, decoder_input_ids]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
