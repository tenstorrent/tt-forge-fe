# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import pytest
import tensorflow as tf
import numpy as np
import torch

import forge
from forge.tensor import to_pt_tensors
from forge.op.eval.common import compare_with_golden_pcc
from forge.config import _get_global_compiler_config
from forge._C import DataFormat

from keras.preprocessing import image


def test_resnet_inference():
    _get_global_compiler_config().default_df_override = DataFormat.Float16_b
    # Load ResNet50 model
    framework_model = tf.keras.applications.ResNet50(weights="imagenet")
    
    input_image = tf.random.uniform((1, 224, 224, 3), dtype=tf.bfloat16)

    # Sanity run
    fw_out = to_pt_tensors(framework_model(input_image))[0]

    # Compile the model
    compiled_model = forge.compile(framework_model, input_image)

    # Execute on device
    co_out = compiled_model(input_image, force_dtype=torch.bfloat16)[0]

    # Compare
    co_out = co_out.to("cpu").to(fw_out.dtype)
    assert compare_with_golden_pcc(golden=fw_out, calculated=co_out.reshape(fw_out.shape), pcc=0.95)


def test_resnet_example():
    _get_global_compiler_config().default_df_override = DataFormat.Float16_b
    # Load ResNet50 model
    framework_model = tf.keras.applications.ResNet50(weights="imagenet")

    cow_image = tf.keras.utils.load_img("forge/test/mlir/resnet/cow.jpg", target_size = (224, 224))
    cow_preprocessed = tf.expand_dims(
        tf.cast(
            tf.keras.applications.resnet.preprocess_input(np.array(cow_image)),
            dtype=tf.bfloat16),
        0
    )

    dog_image = tf.keras.utils.load_img("forge/test/mlir/resnet/dog.png", target_size = (224, 224))
    dog_preprocessed = tf.expand_dims(
        tf.cast(
            tf.keras.applications.resnet.preprocess_input(np.array(dog_image)),
            dtype=tf.bfloat16),
        0
    )
    # Get golden predictions
    tf_cow_logits = to_pt_tensors(framework_model(cow_preprocessed))[0].detach()
    tf_dog_logits = to_pt_tensors(framework_model(dog_preprocessed))[0].detach()
    
    tf_cow_pred = tf.keras.applications.resnet.decode_predictions(tf_cow_logits.to(torch.float32).numpy(), top=1)
    tf_dog_pred = tf.keras.applications.resnet.decode_predictions(tf_dog_logits.to(torch.float32).numpy(), top=1)

    # Compile the model
    compiled_model = forge.compile(framework_model, cow_preprocessed)

    # Execute on device - note that the outputs here will be pytorch tensors
    cow_logits = compiled_model(cow_preprocessed, force_dtype=torch.bfloat16)[0]
    dog_logits = compiled_model(dog_preprocessed, force_dtype=torch.bfloat16)[0]

    cow_pred = tf.keras.applications.resnet.decode_predictions(cow_logits.to(torch.float32).numpy(), top=1)
    dog_pred = tf.keras.applications.resnet.decode_predictions(dog_logits.to(torch.float32).numpy(), top=1)

    print("TENSORFLOW PREDICTION:")
    print(f"For the cow image, predicted: \"{tf_cow_pred[0][0][1]}\" with confidence {tf_cow_pred[0][0][2]}")
    print(f"For the dog image, predicted: \"{tf_dog_pred[0][0][1]}\" with confidence {tf_dog_pred[0][0][2]}")
    print()
    print("TENSTORRENT PREDICTION:")
    print(f"For the cow image, predicted: \"{cow_pred[0][0][1]}\" with confidence {cow_pred[0][0][2]}")
    print(f"For the dog image, predicted: \"{dog_pred[0][0][1]}\" with confidence {dog_pred[0][0][2]}")
