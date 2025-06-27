# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorflow.keras.applications import ResNet50

import forge
from forge.forge_property_utils import Framework, Source, Task, record_model_properties, ModelArch
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify
from test.models.tensorflow.vision.resnet.model_utils.image_utils import get_sample_inputs


# @pytest.mark.push
# @pytest.mark.nightly
# def test_resnet_tensorflow():
#
#     # Record model details
#     module_name = record_model_properties(
#         framework=Framework.TENSORFLOW,
#         model=ModelArch.RESNET,
#         variant="resnet50",
#         source=Source.KERAS,
#         task=Task.IMAGE_CLASSIFICATION,
#     )
#
#     # Load Resnet50 Model
#     framework_model = ResNet50(weights="imagenet")
#
#     # Load sample inputs
#     sample_input = get_sample_inputs()
#     inputs = [sample_input]
#
#     # Compile model
#     compiled_model = forge.compile(framework_model, inputs, module_name=module_name)
#
#     # Verify data on sample input
#     verify(
#         inputs,
#         framework_model,
#         compiled_model,
#         VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.98)),
#     )
