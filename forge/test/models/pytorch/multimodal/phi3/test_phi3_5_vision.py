# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

import forge
from forge.verify.verify import verify

from test.models.pytorch.multimodal.phi3.utils.utils import load_input
from test.models.utils import Framework, Source, Task, build_module_name
from test.utils import download_model


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, pixel_values, image_sizes):
        return self.model(input_ids, attention_mask, None, None, None, pixel_values, image_sizes)


variants = ["microsoft/Phi-3.5-vision-instruct"]


@pytest.mark.nightly
@pytest.mark.xfail(
    reason="NotImplementedError: The following operators are not implemented: ['aten::resolve_neg', 'aten::resolve_conj']"
)
@pytest.mark.parametrize("variant", variants)
def test_phi3_5_vision(forge_property_recorder, variant):

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH,
        model="phi3_5_vision",
        variant=variant,
        task=Task.MULTIMODAL_TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("priority")
    forge_property_recorder.record_model_name(module_name)

    # Load model and processor
    model = download_model(
        AutoModelForCausalLM.from_pretrained,
        variant,
        return_dict=False,
        trust_remote_code=True,
        use_cache=False,
        _attn_implementation="eager",
    )
    model.eval()
    framework_model = Wrapper(model)
    processor = download_model(AutoProcessor.from_pretrained, variant, trust_remote_code=True, num_crops=4)

    # prepare input
    inputs = load_input(processor)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
