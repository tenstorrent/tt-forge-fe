# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO
from PIL import Image
from datasets import load_dataset

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def get_sample_inputs():
    dataset = load_dataset("cifar10", split="test")
    resp = dataset[0]["img"]
    img = Image.open(BytesIO(resp.content)).convert("RGB").resize((224, 224))
    sample_input = img_to_array(img)
    sample_input = np.expand_dims(sample_input, axis=0)
    sample_input = preprocess_input(sample_input)
    sample_input = tf.convert_to_tensor(sample_input, dtype=tf.float32)
    return sample_input
