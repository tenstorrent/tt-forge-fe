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
from transformers import SegformerConfig
from transformers import SegformerForImageClassification

# Forge modules
import forge
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from forge.config import CompilerConfig, MLIRConfig
from forge._C import DataFormat


# Common constants

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
    (512, 512),
]

# Channel size configurations
CHANNEL_SIZE = [
    3,
]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]

# Variants for image classification
VARIANTS = [
    "nvidia/mit-b0",
]


@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("variant", VARIANTS, ids=[f"variant={item}" for item in VARIANTS])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
def test_segformer(
    training,
    batch_size,
    input_size,
    channel_size,
    loop_count,
    variant,
    data_format,
):
    """
    This function creates a basic Segformer model for image classification task using PyTorch.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    module_name = "Segformer"

    # Create random inputs
    input_sample = [
        torch.randn(
            batch_size,
            channel_size,
            input_size[0],
            input_size[1],
        )
    ]

    if data_format == "bfloat16":
        # Convert input to bfloat16
        input_sample = [input.to(torch.bfloat16) for input in input_sample]

    # Set model configurations
    config = SegformerConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = SegformerConfig(**config_dict)

    # Load the model from HuggingFace
    framework_model = SegformerForImageClassification.from_pretrained(variant, config=config)
    if data_format == "bfloat16":
        # Convert model to bfloat16
        framework_model = framework_model.to(torch.bfloat16)
    framework_model.eval()

    # Compiler configuration
    compiler_config = CompilerConfig()
    # Turn on MLIR optimizations.
    compiler_config.mlir_config = (
        MLIRConfig().set_enable_optimizer(True).set_enable_memory_layout_analysis(False).set_enable_fusing(True)
    )
    if data_format == "bfloat16":
        # Convert model to bfloat16
        compiler_config.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=input_sample, module_name=module_name, compiler_cfg=compiler_config
    )

    # Enable program cache on all devices
    settings = DeviceSettings()
    settings.enable_program_cache = True
    configure_devices(device_settings=settings)

    # Run for the first time to warm up the model, it will be done by verify function.
    # This is required to get accurate performance numbers.
    verify(input_sample, framework_model, compiled_model)
    start = time.time()
    for _ in range(loop_count):
        co_out = compiled_model(*input_sample)
    end = time.time()

    fw_out = framework_model(*input_sample)[0]
    co_out = [co.to("cpu") for co in co_out]
    AutomaticValueChecker().check(fw_out=fw_out, co_out=co_out[0])

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    model_name = "Segformer"
    model_type = "Classification, Random Input Data"
    dataset_name = "Segformer, Random Data"
    num_layers = 54  # Number of layers in the model, in this case number of convolutional layers

    print("====================================================================")
    print("| Segformer Benchmark Results:                                     |")
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


def segformer_benchmark(config: dict):
    """
    Run the segformer benchmark.
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

    result = test_segformer(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
        variant=variant,
        data_format=data_format,
    )

    if not output_file:
        output_file = f"forge-benchmark-e2e-segformer_{result['run_type']}.json"
    result["output"] = output_file

    # Save the results to a file
    with open(result["output"], "w") as f:
        json.dump(result, f)
