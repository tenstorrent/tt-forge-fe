# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# From: https://huggingface.co/alibaba-damo/mgp-str-base
import pytest
import torch

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import DepricatedVerifyConfig, verify

from test.models.pytorch.vision.mgp_str_base.utils.utils import load_input, load_model


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs).logits


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("alibaba-damo/mgp-str-base", marks=[pytest.mark.xfail]),
    ],
)
def test_mgp_scene_text_recognition(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="mgp",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.SCENE_TEXT_RECOGNITION,
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load model and input
    framework_model = load_model(variant)
    framework_model = Wrapper(framework_model)
    inputs = load_input(variant)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        verify_cfg=DepricatedVerifyConfig(verify_forge_codegen_vs_framework=True),
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
