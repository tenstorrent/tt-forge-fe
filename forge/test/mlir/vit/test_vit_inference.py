# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import forge
import pytest

from test.mlir.vit.utils.utils import load_model


@pytest.mark.parametrize("model_path", ["google/vit-base-patch16-224"])
def test_vit_inference(model_path):

    # Load Vision Transformer (ViT) model
    framework_model, image_processor = load_model(model_path=model_path)

    # Prepare input
    input_image = torch.rand(1, 3, 224, 224)  # Simulated image tensor
    inputs = image_processor(images=input_image, return_tensors="pt").pixel_values

    # Sanity run
    fw_out = framework_model(inputs)

    # Compile the model
    compiled_model = forge.compile(framework_model, inputs)
    co_out = compiled_model(inputs)

    # TODO: add verification
