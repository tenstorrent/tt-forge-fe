# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorflow.keras.applications import ResNet50

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify
from test.models.tensorflow.vision.resnet.utils.image_utils import get_sample_inputs


@pytest.mark.push
@pytest.mark.nightly
def test_resnet_tensorflow(forge_property_recorder):

    # Record model details
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.TENSORFLOW,
        model="resnet",
        variant="resnet50",
        source=Source.KERAS,
        task=Task.IMAGE_CLASSIFICATION,
    )

    forge_property_recorder.record_group("generality")

    # Load Resnet50 Model
    framework_model = ResNet50(weights="imagenet")

    # Load sample inputs
    sample_input = get_sample_inputs()
    inputs = [sample_input]

    # Compile model
    compiled_model = forge.compile(
        framework_model, inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Verify data on sample input
    verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98)),
        forge_property_handler=forge_property_recorder,
    )
