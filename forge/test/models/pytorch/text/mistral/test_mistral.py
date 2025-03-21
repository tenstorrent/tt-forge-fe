# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralConfig

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, Source, Task, build_module_name

variants = ["mistralai/Mistral-7B-v0.1"]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_mistral(forge_property_recorder, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 30 GB)")

    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="mistral", variant=variant, task=Task.CAUSAL_LM, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    configuration = MistralConfig()
    configuration.sliding_window = None
    configuration.use_cache = False
    configuration.return_dict = False

    framework_model = AutoModelForCausalLM.from_pretrained(variant, device_map="auto", config=configuration)
    tokenizer = AutoTokenizer.from_pretrained(variant)

    framework_model.eval()

    # test should work for batch size 1 and seqlen <= 128
    # for larger seqlen, a DRAM allocation problem might occur (this model is already near maximum model size for single chip)
    prompt = "Of course, fancy writing doesn't just conceal ideas. It can also conceal the lack of them. That's why some people write that way, to conceal the fact that they have nothing to say. Whereas writing simply keeps you honest. If you say nothing simply, it will be obvious to everyone, including you. Simple writing also lasts better. People reading your stuff in the future will be in much the same position as people from other countries reading it today. The culture and the language will have changed. It's not vain to care about that, any more than it's vain for "
    sample_inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    inputs = [sample_inputs]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        inputs,
        module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
