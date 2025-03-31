# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import paddle
import pytest

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from paddle.vision.models import mobilenet_v2

from test.models.utils import Framework, Source, Task, build_module_name

@pytest.mark.nightly
def test_mobilenetv2_basic(forge_property_recorder):
    # Record model details
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="mobilenetv2",
        variant="basic",
        source=Source.PADDLE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    forge_property_recorder.record_model_name(module_name)

    # Load framework model
    framework_model = mobilenet_v2(pretrained=True)

    # Compile model
    input_sample = [paddle.rand([1, 3, 224, 224])]
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=input_sample,
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
    )

    # Verify data on sample input
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
        forge_property_handler=forge_property_recorder,
    )