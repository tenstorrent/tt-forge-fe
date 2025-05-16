# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify
from third_party.tt_forge_models.yolov4 import ModelLoader


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        x, y, z = self.model(image)
        # Post processing inside model casts output to float32,
        # even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return x.to(image.dtype), y.to(image.dtype), z.to(image.dtype)


# @pytest.mark.xfail
@pytest.mark.nightly
def test_yolo_v4(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="Yolo v4",
        variant="default",
        task=Task.OBJECT_DETECTION,
        source=Source.GITHUB,
    )
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P1")

    # Load model and input
    framework_model = ModelLoader.load_model()
    framework_model = Wrapper(framework_model)
    input_sample = ModelLoader.load_inputs()

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=[input_sample],
        module_name=module_name,
        forge_property_handler=forge_property_recorder,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    verify([input_sample], framework_model, compiled_model, forge_property_handler=forge_property_recorder)
