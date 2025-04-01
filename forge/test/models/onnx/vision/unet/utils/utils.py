# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import urllib
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Normalize
import numpy as np


def load_inputs(url, filename):
    try:
        # Download image
        urllib.request.urlretrieve(url, filename)
        input_image = Image.open(filename)
        input_array = np.array(input_image)
        m, s = np.mean(input_array, axis=(0, 1)), np.std(input_array, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(input_image)
        img_batch = input_tensor.unsqueeze(0)
        return [img_batch]

    except Exception as e:
        random_data = torch.randn(1, 3, 256, 256)
        return [random_data]
