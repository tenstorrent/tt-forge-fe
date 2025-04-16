# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
from PIL import Image

import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify

# sys.path.append("forge/test/model_demos/models")
# from fchardnet import get_model, fuse_bn_recursively


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_fchardnet(forge_property_recorder):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH, model="fchardnet", task=Task.IMAGE_CLASSIFICATION, source=Source.TORCHVISION
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")

    # Load and pre-process image
    image_path = "tt-forge-fe/forge/test/model_demos/high_prio/cnn/pytorch/model2/pytorch/pidnet/image/road_scenes.png"
    img = Image.open(image_path)
    img = np.array(img.resize((320, 320)), dtype=np.uint8)
    img = img[:, :, ::-1]
    mean = np.array([0.406, 0.456, 0.485]) * 255
    std = np.array([0.225, 0.224, 0.229]) * 255
    img = (img.astype(np.float64) - mean) / std
    img = torch.tensor(img).float().permute(2, 0, 1)
    input_image = img.unsqueeze(0)

    # Load model
    device = torch.device("cpu")
    arch = {"arch": "hardnet"}
    framework_model = get_model(arch, 19).to(device)
    framework_model = fuse_bn_recursively(model)
    framework_model.eval()

    inputs = [input_image]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
