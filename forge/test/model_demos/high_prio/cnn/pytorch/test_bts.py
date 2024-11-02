# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torchvision import transforms
import numpy as np
import forge

from PIL import Image
import pytest
import os

# import sys

# sys.path = list(
#     set(sys.path + ["third_party/confidential_customer_models/internal/bts/"])
# )

# from scripts.model import get_bts_model
variants = ["densenet161_bts", "densenet121_bts"]


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.parametrize("variant", variants, ids=variants)
def test_bts_pytorch(test_device, variant):

    # Set PyBuda configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.INIT_COMPILE

    # Load sample image
    image_path = "third_party/confidential_customer_models/internal/bts/files/samples/rgb_00315.jpg"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = normalize(image)
    image = torch.unsqueeze(image, 0)

    # Get the model
    model = get_bts_model(variant)
    checkpoint = torch.load(
        "third_party/confidential_customer_models/internal/bts/files/weights/nyu/"
        + str(variant)
        + "/"
        + str(variant)
        + ".pt",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint)
    model.eval()

    class BtsModel_wrapper(torch.nn.Module):
        def __init__(self, model, focal):
            super().__init__()
            self.model = model
            self.focal = focal

        def forward(self, input_tensor):
            return self.model(input_tensor, self.focal)

    bts_model_wrapper = BtsModel_wrapper(model, focal=518.8579)
    bts_model_wrapper.eval()

    inputs = [image]

    compiled_model = forge.compile(bts_model_wrapper, sample_inputs=inputs, module_name="pt_" + str(variant))
