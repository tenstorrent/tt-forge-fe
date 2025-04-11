# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from transformers import FlaxResNetForImageClassification
import jax.random as random

import forge
from forge.verify import verify
from forge.forge_property_utils import Framework, Source, Task

from test.utils import download_model


variants = [
    "microsoft/resnet-50",
]


# TODO: Remove xfail once the unsupported broadcast operation issue is resolved.
@pytest.mark.xfail
@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_resnet(forge_property_recorder, variant):

    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.JAX,
        model="resnet",
        variant="50",
        source=Source.HUGGINGFACE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    forge_property_recorder.record_group("generality")

    framework_model = download_model(FlaxResNetForImageClassification.from_pretrained, variant, return_dict=False)

    key = random.PRNGKey(0)
    input_sample = [random.normal(key, shape=(1, 3, 224, 224))]

    # TODO: The constant folding pass causes the graph to break around the multiply operation at the end.
    # Therefore, this pass is skipped until the issue is resolved.
    # ref: https://github.com/tenstorrent/tt-forge-fe/issues/1709
    os.environ["FORGE_DISABLE_CONSTANT_FOLDING"] = "1"
    compiled_model = forge.compile(
        framework_model, input_sample, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    verify(
        input_sample,
        framework_model,
        compiled_model,
        forge_property_handler=forge_property_recorder,
    )
