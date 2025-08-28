# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.phi3.phi_3_5.pytorch import ModelLoader as Phi35Loader
from third_party.tt_forge_models.phi3.phi_3_5.pytorch import (
    ModelVariant as Phi35Variant,
)

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

variants = [Phi35Variant.MINI_INSTRUCT]


@pytest.mark.out_of_memory
@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants)
def test_phi3_5_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3_5,
        variant=variant.value,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

<<<<<<< HEAD
    pytest.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2903")

    # Load model and inputs via loader
    loader = Phi35Loader(variant)
    framework_model = loader.load_model()
    framework_model.eval()
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


variants = [Phi35Variant.MOE_INSTRUCT]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_phi3_5_moe_causal_lm(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.PHI3_5_MOE,
        variant=variant.value,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    pytest.xfail(reason="Requires multi-chip support")
