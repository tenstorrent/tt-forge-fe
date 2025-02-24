# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

import forge
from forge.verify.verify import verify

from test.models.utils import Framework, build_module_name

# import sys

# sys.path = list(
#     set(sys.path + ["third_party/confidential_customer_models/internal/bts/"])
# )

# from scripts.model import get_bts_model
variants = ["densenet161_bts", "densenet121_bts"]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.nightly
def test_bts_pytorch(record_forge_property, variant):
    if variant != "densenet161_bts":
        pytest.skip("Skipping due to the current CI/CD pipeline limitations")

    # Build Module Name
    module_name = build_module_name(framework=Framework.PYTORCH, model="bts", variant=variant)

    # Record Forge Property
    record_forge_property("model_name", module_name)

    # Load sample image
    image_path = "third_party/confidential_customer_models/internal/bts/files/samples/rgb_00315.jpg"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = normalize(image)
    image = torch.unsqueeze(image, 0)

    # Get the model
    framework_model = get_bts_model(variant)
    checkpoint = torch.load(
        "third_party/confidential_customer_models/internal/bts/files/weights/nyu/"
        + str(variant)
        + "/"
        + str(variant)
        + ".pt",
        map_location=torch.device("cpu"),
    )
    framework_model.load_state_dict(checkpoint)
    framework_model.eval()

    class BtsModel_wrapper(torch.nn.Module):
        def __init__(self, model, focal):
            super().__init__()
            self.model = model
            self.focal = focal

        def forward(self, input_tensor):
            return self.model(input_tensor, self.focal)

    framework_model = BtsModel_wrapper(framework_model, focal=518.8579)
    framework_model.eval()

    inputs = [image]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
