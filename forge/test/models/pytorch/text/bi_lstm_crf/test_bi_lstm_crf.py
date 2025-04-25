# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.text.bi_lstm_crf.utils.model import get_model


@pytest.mark.nightly
@pytest.mark.xfail
def test_birnn_crf(forge_property_recorder):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="BiRnnCrf",
        task=Task.TOKEN_CLASSIFICATION,
        source=Source.GITHUB,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P1")

    test_sentence = ["apple", "corporation", "is", "in", "georgia"]

    # Load model and input tensor
    model, test_input = get_model(test_sentence)
    model.eval()

    # Forge compile framework model
    compiled_model = forge.compile(
        model, sample_inputs=(test_input,), module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(test_input, model, compiled_model, forge_property_handler=forge_property_recorder)
