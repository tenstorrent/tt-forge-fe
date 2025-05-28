# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import random

import paddle
import pytest
from datasets import load_dataset

from paddle.vision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from forge.forge_property_utils import Framework, Source, Task, record_model_properties

variants = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_resnet_pd(variant):
    # Record model details
    module_name = record_model_properties(
        framework=Framework.PADDLE,
        model="resnet",
        variant=variant[6:],
        source=Source.PADDLE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load framework model
    framework_model = eval(variant)(pretrained=True)

    # Compile model
    input_sample = [paddle.rand([1, 3, 224, 224])]
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input_sample,
        module_name=module_name,
    )

    # Verify data on sample input
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )
