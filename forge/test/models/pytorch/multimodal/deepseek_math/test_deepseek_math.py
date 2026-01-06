# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.deepseek.deepseek_math.pytorch import (
    ModelLoader as MathLoader,
)
from third_party.tt_forge_models.deepseek.deepseek_math.pytorch import (
    ModelVariant as MathVariant,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify

from test.models.models_utils import generate_no_cache, pad_inputs
from test.models.pytorch.multimodal.deepseek_math.model_utils.model_utils import (
    DeepSeekWrapper,
)

DEEPSEEK_MATH_VARIANTS = [
    MathVariant.DEEPSEEK_7B_INSTRUCT,
]


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", DEEPSEEK_MATH_VARIANTS)
def test_deepseek_math_inference_no_cache(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DEEPSEEK,
        variant=variant,
        task=Task.NLP_QA,
        source=Source.HUGGINGFACE,
    )
    pytest.xfail(reason="Requires multi-chip support")

    # Load Model and Tokenizer
    loader = MathLoader(variant=variant)
    model = loader.load_model()
    tokenizer = loader._load_tokenizer()
    framework_model = DeepSeekWrapper(model)
    framework_model.eval()

    # Prepare inputs
    inputs = loader.load_inputs()
    padded_inputs, seq_len = pad_inputs(inputs)

    # Compile model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[padded_inputs],
        module_name=module_name,
    )

    # Verify correctness
    verify([padded_inputs], framework_model, compiled_model)

    # Generate output
    generated_text = generate_no_cache(
        max_new_tokens=512,
        model=compiled_model,
        inputs=padded_inputs,
        seq_len=seq_len,
        tokenizer=tokenizer,
    )
    print(generated_text)
