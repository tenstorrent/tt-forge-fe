# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import onnx
import requests
import os
import torch
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
import json
import torch.nn.functional as F
from tabulate import tabulate


def load_model(variant):
    onnx_dir_path = "mobilenetv2"
    onnx_model_path = f"mobilenetv2/{variant}_Opset17.onnx"
    if not os.path.exists(onnx_model_path):
        if not os.path.exists("mobilenetv2"):
            os.mkdir("mobilenetv2")
        url = f"https://github.com/onnx/models/raw/main/Computer_Vision/{variant}_Opset17_timm/{variant}_Opset17.onnx?download="
        response = requests.get(url, stream=True)
        with open(onnx_model_path, "wb") as f:
            f.write(response.content)

    model_name = f"mobilenetv2_{variant}_onnx"
    onnx_model = onnx.load(onnx_model_path)
    return onnx_model, onnx_dir_path


def load_inputs(onnx_model):
    dataset = load_dataset("huggingface/cats-image")
    img = dataset["test"]["image"][0]

    input_tensors = []

    for input_tensor in onnx_model.graph.input:
        input_name = input_tensor.name
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        input_height = input_shape[2]
        input_width = input_shape[3]
        transform = transforms.Compose(
            [
                transforms.Resize((input_height, input_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Apply the transformation
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        input_tensors.append(img_tensor)

    return input_tensors


def print_results(fw_out, compiled_model_out):
    fw_softmax_output = F.softmax(fw_out, dim=1)
    compiled_model_softmax_output = F.softmax(compiled_model_out, dim=1)
    fw_predicted_class_idx = torch.argmax(fw_softmax_output, dim=1).item()
    compiled_model_predicted_class_idx = torch.argmax(compiled_model_softmax_output, dim=1).item()
    fw_predicted_class_prob = fw_softmax_output[0, fw_predicted_class_idx].item()
    compiled_model_predicted_class_prob = compiled_model_softmax_output[0, compiled_model_predicted_class_idx].item()

    # Directly load ImageNet class labels inside post-process
    class_labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(class_labels_url)
    class_idx = response.json()

    # Map class indices to human-readable labels
    class_labels = [class_idx[str(i)][1] for i in range(len(class_idx))]
    fw_predicted_class_label = class_labels[fw_predicted_class_idx]
    compiled_model_predicted_class_label = class_labels[compiled_model_predicted_class_idx]
    table = [
        ["Metric", "Framework Model", "Compiled Model"],
        ["Predicted Class Index", fw_predicted_class_idx, compiled_model_predicted_class_idx],
        ["Predicted Class Label", fw_predicted_class_label, compiled_model_predicted_class_label],
        ["Predicted Class Probability", fw_predicted_class_prob, compiled_model_predicted_class_prob],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
