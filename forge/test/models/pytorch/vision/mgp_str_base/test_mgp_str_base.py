# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# From: https://huggingface.co/alibaba-damo/mgp-str-base
import pytest
import torch

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.verify import DepricatedVerifyConfig, verify

from test.models.pytorch.vision.mgp_str_base.model_utils.utils import (
    load_input,
    load_model,
)


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
def test_mgp_scene_text_recognition(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="mgp",
        variant=variant,
        source=Source.HUGGINGFACE,
        task=Task.SCENE_TEXT_RECOGNITION,
    )

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
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model)
