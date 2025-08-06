# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from third_party.tt_forge_models.rcnn.pytorch.loader import ModelLoader

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify


# Paper - https://arxiv.org/abs/1311.2524
# Repo - https://github.com/object-detection-algorithm/R-CNN
@pytest.mark.nightly
def test_rcnn_pytorch():
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH, model=ModelArch.RCNN, source=Source.TORCHVISION, task=Task.OBJECT_DETECTION
    )

    # Load RCNN model using the new loader
    loader = ModelLoader(num_classes=2)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)

    # Generate region proposals and process them one by one (matching original test behavior)
    for idx, inputs in loader.generate_region_proposals_iterator(dtype_override=torch.bfloat16):

        # Record Forge Property for each region
        module_name = record_model_properties(
            framework=Framework.PYTORCH,
            model=ModelArch.RCNN,
            suffix=f"rect_{idx}",
            source=Source.TORCHVISION,
            task=Task.OBJECT_DETECTION,
        )

        data_format_override = DataFormat.Float16_b
        compiler_cfg = CompilerConfig(default_df_override=data_format_override)

        # Forge compile framework model
        compiled_model = forge.compile(
            framework_model,
            sample_inputs=inputs,
            module_name=module_name,
            compiler_cfg=compiler_cfg,
        )

        verify(inputs, framework_model, compiled_model)

        break  # As generated proposals will be around 2000, halt inference after getting result from single proposal.
