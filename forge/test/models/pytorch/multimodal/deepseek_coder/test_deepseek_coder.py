# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.multimodal.deepseek_coder.utils.model_utils import (
    DeepSeekWrapper,
    download_model_and_tokenizer,
    generate_no_cache,
    pad_inputs,
)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["deepseek-coder-1.3b-instruct"])
def test_deepseek_inference_no_cache(forge_property_recorder, variant):
    pytest.skip("Insufficient host DRAM to run this model (requires a bit more than 32 GB during compile time)")

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="deepseek", variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )

    # Record Forge Property
    forge_property_recorder("model_name", module_name)

    # Load Model and Tokenizer
    model_name = f"deepseek-ai/{variant}"
    model, tokenizer, inputs = download_model_and_tokenizer(model_name)
    framework_model = DeepSeekWrapper(model)
    framework_model.eval()

    padded_inputs, seq_len = pad_inputs(inputs)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model, forge_property_handler=forge_property_recorder)

    generated_text = generate_no_cache(
        max_new_tokens=512, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )
    print(generated_text)


@pytest.mark.parametrize("variant", ["deepseek-coder-1.3b-instruct"])
def test_deepseek_inference_no_cache_cpu(variant):
    model_name = f"deepseek-ai/{variant}"
    model, tokenizer, inputs = download_model_and_tokenizer(model_name)

    framework_model = DeepSeekWrapper(model)
    framework_model.eval()

    padded_inputs, seq_len = pad_inputs(inputs)

    generated_text = generate_no_cache(
        max_new_tokens=512, model=framework_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )
    print(generated_text)
