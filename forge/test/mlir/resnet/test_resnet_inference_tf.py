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
from prettytable import PrettyTable

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

    pizza_image = tf.keras.utils.load_img("forge/test/mlir/resnet/pizza.jpg", target_size = (224, 224))
    pizza_preprocessed = tf.expand_dims(
        tf.cast(
            tf.keras.applications.resnet.preprocess_input(np.array(pizza_image)),
            dtype=tf.bfloat16),
        0
    )

    hammer_image = tf.keras.utils.load_img("forge/test/mlir/resnet/hammer.jpeg", target_size = (224, 224))
    hammer_preprocessed = tf.expand_dims(
        tf.cast(
            tf.keras.applications.resnet.preprocess_input(np.array(hammer_image)),
            dtype=tf.bfloat16),
        0
    )
    # Get golden predictions
    tf_pizza_logits = to_pt_tensors(framework_model(pizza_preprocessed))[0].detach()
    tf_hammer_logits = to_pt_tensors(framework_model(hammer_preprocessed))[0].detach()
    
    tf_pizza_pred = tf.keras.applications.resnet.decode_predictions(tf_pizza_logits.to(torch.float32).numpy(), top=5)
    tf_hammer_pred = tf.keras.applications.resnet.decode_predictions(tf_hammer_logits.to(torch.float32).numpy(), top=5)

    # Compile the model                           Sample input for trace
    compiled_model = forge.compile(framework_model, pizza_preprocessed)

    # Execute on device - note that the outputs here will be pytorch tensors
    pizza_logits = compiled_model(pizza_preprocessed, force_dtype=torch.bfloat16)[0]
    hammer_logits = compiled_model(hammer_preprocessed, force_dtype=torch.bfloat16)[0]

    pizza_pred = tf.keras.applications.resnet.decode_predictions(pizza_logits.to(torch.float32).numpy(), top=5)
    hammer_pred = tf.keras.applications.resnet.decode_predictions(hammer_logits.to(torch.float32).numpy(), top=5)

    table = [['TENSORFLOW PREDICTION', 'TENSTORRENT PREDICTION']]
    tab = PrettyTable(table[0])
    tab.add_row(['Pizza image', 'Pizza Image'], divider=True)

    tf_pizza_pred_str = []
    for pred in tf_pizza_pred[0]:
        tf_pizza_pred_str.append(f"{pred[1]}: {pred[2]}")
    tf_pizza_pred_str = "\n".join(tf_pizza_pred_str)

    pizza_pred_str = []
    for pred in pizza_pred[0]:
        pizza_pred_str.append(f"{pred[1]}: {pred[2]}")
    pizza_pred_str = "\n".join(pizza_pred_str)
    tab.add_row([tf_pizza_pred_str, pizza_pred_str], divider=True)

    tab.add_row(["Hammer image", "Hammer image"], divider=True)

    tf_hammer_pred_str = []
    for pred in tf_hammer_pred[0]:
        tf_hammer_pred_str.append(f"{pred[1]}: {pred[2]}")
    tf_hammer_pred_str = "\n".join(tf_hammer_pred_str)

    hammer_pred_str = []
    for pred in hammer_pred[0]:
        hammer_pred_str.append(f"{pred[1]}: {pred[2]}")
    hammer_pred_str = "\n".join(hammer_pred_str)
    tab.add_row([tf_hammer_pred_str, hammer_pred_str])

    print(tab)