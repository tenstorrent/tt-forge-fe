# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import forge
from test.mlir.vit.utils.utils import load_model
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig


@pytest.mark.parametrize("model_path", ["google/vit-base-patch16-224"])
def test_vit_inference(model_path):
    # Load Vision Transformer (ViT) model
    framework_model, image_processor = load_model(model_path=model_path)

    # Prepare input
    input_image = torch.rand(1, 3, 224, 224)  # Simulated image tensor
    inputs = image_processor(images=input_image, return_tensors="pt").pixel_values

    # Compile the model
    compiled_model = forge.compile(framework_model, inputs)

    # Run inference and verify the output
    verify(inputs, framework_model, compiled_model, VerifyConfig(verify_data=False, verify_allclose=False))
