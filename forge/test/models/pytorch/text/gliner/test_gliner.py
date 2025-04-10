# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from gliner import GLiNER

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.text.gliner.utils.model_utils import (
    GlinerWrapper,
    post_processing,
    pre_processing,
)

variants = ["urchade/gliner_multi-v2.1"]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_gliner(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="Gliner",
        variant=variant,
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P1")

    # Load model
    model = GLiNER.from_pretrained(variant)

    # prepare input
    text = """
    Cristiano Ronaldo dos Santos Aveiro was born 5 February 1985) is a Portuguese professional footballer.
    """
    labels = ["person", "award", "date", "competitions", "teams"]
    inputs, raw_batch = pre_processing(model, [text], labels)

    # Forge compile framework model
    framework_model = GlinerWrapper(model)
    framework_model.eval()
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    fw_out, co_out = verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    # Post processing
    entities = post_processing(model, co_out, [text], raw_batch)
    for entity in entities:
        print(entity["text"], "=>", entity["label"])
