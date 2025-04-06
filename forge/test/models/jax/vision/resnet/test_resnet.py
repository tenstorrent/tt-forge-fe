# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from transformers import FlaxResNetForImageClassification
import jax.random as random

import forge
from forge.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker

from test.models.utils import Framework, Source, Task, build_module_name


variants = [
    "microsoft/resnet-50",
]


# TODO: Remove xfail once the unsupported broadcast operation issue is resolved.
@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_resnet(forge_property_recorder, variant):

    module_name = build_module_name(
        framework=Framework.JAX,
        model="resnet",
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    framework_model = FlaxResNetForImageClassification.from_pretrained(variant)

    key = random.PRNGKey(0)
    input_sample = [random.normal(key, shape=(1, 3, 224, 224))]

    # TODO: The constant folding pass causes the graph to break around the multiply operation at the end.
    # Therefore, this pass is skipped until the issue is resolved.
    # ref: https://github.com/tenstorrent/tt-forge-fe/issues/1709
    os.environ["FORGE_DISABLE_CONSTANT_FOLDING"] = "1"
    compiled_model = forge.compile(
        framework_model, sample_inputs=input_sample, forge_property_handler=forge_property_recorder
    )

    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.99)),
        forge_property_handler=forge_property_recorder,
    )
