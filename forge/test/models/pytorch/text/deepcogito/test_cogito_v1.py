# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

from test.models.pytorch.text.deepcogito.utils.model import get_input_model


@pytest.mark.skip("Skipping due to Out of Memory issue")
@pytest.mark.nightly
@pytest.mark.parametrize("variant", ["deepcogito/cogito-v1-preview-llama-3B"])
def test_cogito_generation(forge_property_recorder, variant):

    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="cogito",
        variant=variant,
        task=Task.TEXT_GENERATION,
        source=Source.HUGGINGFACE,
    )
    forge_property_recorder.record_group("generality")

    # Load model and tokenizer
    input_tensor_list, framework_model = get_input_model(variant)

    # Compile with Forge
    compiled_model = forge.compile(
        framework_model,
        input_tensor_list,
        module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Run verification
    verify(input_tensor_list, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
