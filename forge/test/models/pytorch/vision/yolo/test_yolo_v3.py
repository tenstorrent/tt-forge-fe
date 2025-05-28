# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task, record_model_properties
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify

from third_party.tt_forge_models.yolov3 import ModelLoader  # isort:skip


@pytest.mark.nightly
def test_yolo_v3():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model="Yolo v3",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )

    # Load model and input
    framework_model = ModelLoader.load_model()
    input_sample = ModelLoader.load_inputs()

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[input_sample],
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify(
        [input_sample],
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )
