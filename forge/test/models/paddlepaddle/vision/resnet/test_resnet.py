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

from test.models.utils import Framework, Source, Task, build_module_name

@pytest.mark.push
@pytest.mark.nightly
def test_resnet_pd(forge_property_recorder):
    random.seed(0)

    # Record model details
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="resnet",
        variant="50",
        source=Source.PADDLE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    forge_property_recorder.record_model_name(module_name)

    # Load tiny dataset
    dataset = load_dataset("zh-plus/tiny-imagenet")
    images = random.sample(dataset["valid"]["image"], 10)

    # Load framework model
    framework_model = resnet50(pretrained=True)

    # Compile model
    input_sample = [paddle.rand([1, 3, 224, 224])]
    compiled_model = forge.compile(framework_model, input_sample, forge_property_handler=forge_property_recorder)

    # Verify data on sample input
    verify(input_sample, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)), forge_property_handler=forge_property_recorder)


variants = [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152',
]
@pytest.mark.parametrize("variant", variants)
@pytest.mark.nightly
def test_resnet_pd_variants(variant, forge_property_recorder):
    random.seed(0)

    # Record model details
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="resnet",
        variant=variant,
        source=Source.PADDLE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    forge_property_recorder.record_model_name(module_name)

    # Load framework model
    framework_model = eval(variant)(pretrained=True)

    # Compile model
    input_sample = [paddle.rand([1, 3, 224, 224])]
    compiled_model = forge.compile(framework_model, input_sample, forge_property_handler=forge_property_recorder)

    # Verify data on sample input
    verify(input_sample, framework_model, compiled_model, VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)), forge_property_handler=forge_property_recorder)