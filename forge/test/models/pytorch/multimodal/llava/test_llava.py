# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest

import forge
from forge.verify.verify import verify
from utils import load_inputs, load_model

variants = ["llava-hf/llava-1.5-7b-hf"]


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.parametrize("variant", variants)
def test_llava(variant):

    framework_model, processor = load_model(variant)
    image = "https://www.ilankelman.org/stopsigns/australia.jpg"
    text = "What’s shown in this image?"
    input_ids, attn_mask, pixel_values = load_inputs(image, text, processor)
    inputs = [input_ids, attn_mask, pixel_values]

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    verify(inputs, framework_model, compiled_model)
