# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from PIL import Image

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from third_party.tt_forge_models.tools.utils import get_file


def get_sample_inputs():
    input_image = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
    img = Image.open(str(input_image)).convert("RGB").resize((224, 224))
    sample_input = img_to_array(img)
    sample_input = np.expand_dims(sample_input, axis=0)
    sample_input = preprocess_input(sample_input)
    sample_input = tf.convert_to_tensor(sample_input, dtype=tf.float32)
    return sample_input
