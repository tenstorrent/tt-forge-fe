# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import pytest
import time
import socket
import json
from datetime import datetime

# Third-party modules
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoFeatureExtractor, ViTForImageClassification, DeiTConfig

# Forge modules
import forge
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from test.utils import download_model
from forge.config import CompilerConfig, MLIRConfig
from test.benchmark.utils import load_benchmark_dataset, evaluate_classification
from forge._C import DataFormat


# Common constants

# Machine learning task
TASK = [
    "classification",
]

# Batch size configurations
BATCH_SIZE = [
    1,
]

# Data format configurations
DATA_FORMAT = [
    "bfloat16",
]

# Input size configurations
INPUT_SIZE = [
    (224, 224),
]

# Channel size configurations
CHANNEL_SIZE = [
    3,
]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]

# Variants for image classification
VARIANTS = [
    "facebook/deit-tiny-patch16-224",
]


def test_deit_tiny(training, batch_size, input_size, channel_size, loop_count, variant, task, data_format):
    """
    Test the DeiT base benchmark function.
    It is used for benchmarking purposes.
    """

    # sequence_size = 224
    # image_channels = 3
    # image_size = 224

    module_name = "DeiTTiny"

    model_name = "facebook/deit-tiny-patch16-224"
    config = DeiTConfig.from_pretrained(model_name)
    batch_size = 1
    config.num_hidden_layers = 12
    model = ViTForImageClassification.from_pretrained(model_name, config=config)
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image_processor = AutoFeatureExtractor.from_pretrained("facebook/deit-tiny-patch16-224")
    torch_pixel_values = image_processor(image, return_tensors="pt").pixel_values
    torch_pixel_values = torch_pixel_values.repeat(batch_size, 1, 1, 1)

    print("Torch pixel values shape:")
    print(torch_pixel_values.shape)

    compiler_cfg = CompilerConfig()
    compiler_cfg.mlir_config = MLIRConfig().set_enable_consteval(True).set_enable_optimizer(True)

    compiled_model = forge.compile(
        model, sample_inputs=[torch_pixel_values], module_name=module_name, compiler_cfg=compiler_cfg
    )

    # file_path = "generated_export_deit_tiny.cpp"
    # compiled_model.export_to_cpp(file_path)

    # Inference with compiled model
    # output = compiled_model(torch_pixel_values)

    start = time.time()
    for i in tqdm(range(loop_count)):
        co_out = compiled_model(torch_pixel_values)
    end = time.time()

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    model_name = "DeiT Base"
    model_type = "Classification"
    if task == "classification":
        model_type += ", ImageNet-1K"
        dataset_name = "ImageNet-1K"
    elif task == "na":
        model_type += ", Random Input Data"
        dataset_name = model_name + ", Random Data"
    num_layers = 1  # Number of layers in the model, in this case number of convolutional layers

    print("====================================================================")
    print("| DeiT Benchmark Results:                                          |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: {total_time}")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec}")
    print(f"| Batch size: {batch_size}")
    print(f"| Data format: {data_format}")
    print(f"| Input size: {input_size}")
    print(f"| Channel size: {channel_size}")
    print("====================================================================")

    result = {
        "model": model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{num_layers}_{loop_count}",
        "config": {"model_size": "small"},
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": data_format,
        # "math_fidelity": math_fidelity, @TODO - For now, we are skipping these parameters, because we are not supporting them
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "output_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "image_dimension": f"{channel_size}x{input_size[0]}x{input_size[1]}",
        "perf_analysis": False,
        "training": training,
        "measurements": [
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_samples",
                "value": total_samples,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_time",
                "value": total_time,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
        ],
        "device_info": {
            "device_name": "",
            "galaxy": False,
            "arch": "",
            "chips": 1,
        },
        "device_ip": None,
    }

    return result


def deit_tiny_benchmark(config: dict):
    """
    Run the deit benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    data_format = config["data_format"]
    input_size = INPUT_SIZE[0]
    channel_size = CHANNEL_SIZE[0]
    output_file = config["output"]
    loop_count = config["loop_count"]
    variant = VARIANTS[0]
    task = config["task"]

    result = test_deit_tiny(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
        variant=variant,
        task=task,
        data_format=data_format,
    )

    if not output_file:
        output_file = f"forge-benchmark-e2e-deit_tiny_{result['run_type']}.json"
    result["output"] = output_file

    # Save the results to a file
    with open(result["output"], "w") as f:
        json.dump(result, f)
