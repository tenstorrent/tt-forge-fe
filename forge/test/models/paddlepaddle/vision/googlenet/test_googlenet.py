# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import paddle
import pytest

import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from paddle.vision.models import googlenet

from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.xfail(
    reason="RuntimeError: TT_FATAL @ /tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp:37: (input_shape[3] % tt::constants::TILE_WIDTH == 0) || (input_shape[3] == 16) info: Input channels (528) should be padded to nearest TILE_WIDTH (32) or should be 16"
)
@pytest.mark.nightly
def test_googlenet(forge_property_recorder):
    # Record model details
    module_name = build_module_name(
        framework=Framework.PADDLE,
        model="googlenet",
        source=Source.PADDLE,
        task=Task.IMAGE_CLASSIFICATION,
    )
    forge_property_recorder.record_model_name(module_name)

    # Load framework model
    framework_model = googlenet(pretrained=True)

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
