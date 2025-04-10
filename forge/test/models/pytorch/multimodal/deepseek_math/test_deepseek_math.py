# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task

from test.models.pytorch.multimodal.deepseek_math.utils.model_utils import (
    DeepSeekWrapper,
    download_model_and_tokenizer,
    generation,
)


@pytest.mark.parametrize("variant", ["deepseek-math-7b-instruct"])
def test_deepseek_inference_no_cache_cpu(variant):
    model_name = f"deepseek-ai/{variant}"
    model, tokenizer, input_ids = download_model_and_tokenizer(model_name)

    framework_model = DeepSeekWrapper(model)
    framework_model.eval()

    generated_text = generation(
        max_new_tokens=200, compiled_model=framework_model, input_ids=input_ids, tokenizer=tokenizer
    )
    print(generated_text)


@pytest.mark.parametrize("variant", ["deepseek-math-7b-instruct"])
def test_deepseek_inference(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="deepseek", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    model_name = f"deepseek-ai/{variant}"
    model, tokenizer, input_ids = download_model_and_tokenizer(model_name)
    framework_model = DeepSeekWrapper(model)
    framework_model.eval()

    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[input_ids],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )
    generated_text = generation(
        max_new_tokens=1, compiled_model=compiled_model, input_ids=input_ids, tokenizer=tokenizer
    )
    print(generated_text)
