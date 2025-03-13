# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import requests
import torch

import forge
from forge.verify.verify import verify

from test.models.pytorch.vision.ssd300_resnet50.utils.image_utils import prepare_input
from test.models.utils import Framework, Source, Task, build_module_name


@pytest.mark.nightly
@pytest.mark.xfail(
    reason="RuntimeError: Tensor 0 - stride mismatch: expected [270000, 90000, 300, 1], got [3, 1, 900, 3]"
)
def test_pytorch_ssd300_resnet50(forge_property_recorder):
    # Build Module Name
    module_name = build_module_name(
        framework=Framework.PYTORCH, model="ssd300_resnet50", source=Source.TORCH_HUB, task=Task.IMAGE_CLASSIFICATION
    )

    # Record Forge Property
    forge_property_recorder.record_group("generality")
    forge_property_recorder.record_model_name(module_name)

    # STEP 2 : prepare model
    framework_model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", pretrained=False)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
    checkpoint_path = "nvidia_ssdpyt_fp16_190826.pt"

    response = requests.get(url)
    with open(checkpoint_path, "wb") as f:
        f.write(response.content)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    framework_model.load_state_dict(checkpoint["model"])
    framework_model.eval()

    # STEP 3 : prepare input
    img = "http://images.cocodataset.org/val2017/000000397133.jpg"
    HWC = prepare_input(img)
    CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
    batch = np.expand_dims(CHW, axis=0)
    input_batch = torch.from_numpy(batch).float()
    inputs = [input_batch]

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, forge_property_handler=forge_property_recorder
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
