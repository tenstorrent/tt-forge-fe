# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

import forge
from forge.verify.verify import verify

from test.models.pytorch.text.bloom.utils.utils import load_input, load_model
from test.models.utils import Framework, Source, Task, build_module_name


# Wrapper to get around past key values
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, None, attention_mask)
        return output


@pytest.mark.nightly
def test_bloom(record_forge_property):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="bloom",
        source=Source.HUGGINGFACE,
        task=Task.CAUSAL_LM,
    )

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load model and input
    model = load_model()
    framework_model = Wrapper(model)
    inputs = load_input()

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
