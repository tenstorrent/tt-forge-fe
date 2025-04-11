# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import paddle
import pytest

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from paddle.vision.models import mobilenet_v1

from forge.forge_property_utils import Framework, Source, Task


@pytest.mark.nightly
@pytest.mark.skip(reason="Transient failure: Causing seg faults while building Metal kernels")
def test_mobilenetv1_basic(forge_property_recorder):
    # Record model details
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PADDLE,
        model="mobilenetv1",
        variant="basic",
        source=Source.PADDLE,
        task=Task.IMAGE_CLASSIFICATION,
    )

    # Load framework model
    framework_model = mobilenet_v1(pretrained=True)

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
