# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.utils import download_model

variants = ["microsoft/Phi-3.5-mini-instruct"]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.skip(reason="Test skipped due to segmentation fault issue")
@pytest.mark.parametrize("variant", variants)
def test_phi3_5_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3_5,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and tokenizer
    tokenizer = download_model(AutoTokenizer.from_pretrained, variant)
    framework_model = download_model(
        AutoModelForCausalLM.from_pretrained, variant, return_dict=False, trust_remote_code=True, use_cache=False
    )
    framework_model.eval()

    # prepare input
    input_prompt = "Africa is an emerging economy because"
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=256,
        pad_to_max_length=True,
        truncation=True,
    )
    inputs = [inputs["input_ids"], inputs["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
