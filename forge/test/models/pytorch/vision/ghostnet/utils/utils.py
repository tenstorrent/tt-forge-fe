# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import urllib

import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from test.utils import download_model


def load_ghostnet_model(variant):

    # Create Forge module from PyTorch model
    framework_model = download_model(timm.create_model, variant, pretrained=True)
    framework_model.eval()

    # Prepare input
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename)
    data_config = resolve_data_config({}, model=framework_model)
    transforms = create_transform(**data_config, is_training=False)
    img_tensor = transforms(img).unsqueeze(0)

    return framework_model, [img_tensor]
