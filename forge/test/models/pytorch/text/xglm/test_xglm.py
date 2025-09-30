# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from third_party.tt_forge_models.xglm.pytorch import ModelLoader, ModelVariant

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

variants = [
    ModelVariant.XGLM_564M,
    pytest.param(
        ModelVariant.XGLM_1_7B, 
        marks=[pytest.mark.xfail(reason="https://github.com/tenstorrent/tt-forge-fe/issues/2969")]
    ),
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_xglm_causal_lm(variant):

    pcc = 0.99
    if variant == ModelVariant.XGLM_1_7B:
        pcc = 0.95

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.XGLM,
        variant=variant,
        task=Task.CAUSAL_LM,
        source=Source.HUGGINGFACE,
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model()
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(
        inputs, framework_model, compiled_model, verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc))
    )
