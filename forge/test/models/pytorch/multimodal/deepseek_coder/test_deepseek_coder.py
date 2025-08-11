# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from third_party.tt_forge_models.deepseek.deepseek_coder.pytorch import (
    ModelLoader as CausalLMLoader,
)
from third_party.tt_forge_models.deepseek.deepseek_coder.pytorch import (
    ModelVariant as CausalLMVariant,
)

from test.models.pytorch.multimodal.deepseek_coder.model_utils.model_utils import (
    DeepSeekWrapper,
)

DEEPSEEK_VARIANTS = [
    CausalLMVariant.DEEPSEEK_1_3B_INSTRUCT,
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", DEEPSEEK_VARIANTS)
def test_deepseek_inference_no_cache(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.DEEPSEEK, variant=variant, task=Task.QA, source=Source.HUGGINGFACE
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load Model and Tokenizer
    model_name = f"deepseek-ai/{variant}"
    loader = CausalLMLoader()
    model = loader.load_model()
    framework_model = DeepSeekWrapper(model)
    framework_model.eval()
    tokenizer = loader._load_tokenizer()

    padded_inputs, seq_len = loader.load_inputs()

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
    )

    # Model Verification
    verify([padded_inputs], framework_model, compiled_model)

    generated_text = loader.decode_output(
        max_new_tokens=512, model=compiled_model, inputs=padded_inputs, seq_len=seq_len, tokenizer=tokenizer
    )
    print(generated_text)
