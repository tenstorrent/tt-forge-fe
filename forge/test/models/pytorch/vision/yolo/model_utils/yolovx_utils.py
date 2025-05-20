# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from datasets import load_dataset


def get_test_input():
    dataset = load_dataset("huggingface/cats-image", split="test[:1]")
    image = dataset[0]["image"]
    image_np = np.array(image)
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor
