# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time
import socket
import subprocess
import json
import torch
from torch import nn
import forge
from forge.op.eval.common import compare_with_golden_pcc


# Batch size configurations
MNIST_BATCH_SIZE_EXP_RANGE = 7

# Input size configurations
MNIIST_INPUT_SIZE_EXP_RANGE = [5, 7]
MNIIST_INPUT_SIZE_FACTORS = [1, 3, 5, 7]

# Hidden layer size configurations
MNIST_HIDDEN_SIZE_EXP_RANGE = [5, 7]
MNIIST_HIDDEN_SIZE_FACTORS = [1, 3]

MNIST_INPUT_FEATURE_SIZE = 784  # 784 = 28 * 28, default size of MNIST image 
MNIST_OUTPUT_FEATURE_SIZE = 10  # 10 classes in MNIST, default output size
MNIIST_HIDDEN_SIZE = 256        # Hidden layer size, default size

BATCH_SIZE = [2 ** i for i in range(MNIST_BATCH_SIZE_EXP_RANGE)]    # Batch size, sizes will be 1, 2, 4, 8, 16, 32, 64, etc.
INPUT_SIZE = [     # Input size, sizes will be 1 * 2^5 = 32, 3 * 2^5 = 96, 5 * 2^5 = 160, 7 * 2^5 = 224, etc.
    factor * hidden 
    for factor in MNIIST_INPUT_SIZE_FACTORS 
    for hidden in [2 ** i for i in range(MNIIST_INPUT_SIZE_EXP_RANGE[0], MNIIST_INPUT_SIZE_EXP_RANGE[1])]
]
HIDDEN_SIZE = [     # Hidden layer size, sizes will be 1 * 2^5 = 32, 3 * 2^5 = 96, 1 * 2^6 = 64, 3 * 2^6 = 192, etc.
    factor * hidden 
    for factor in MNIIST_HIDDEN_SIZE_FACTORS 
    for hidden in [2 ** i for i in range(MNIST_HIDDEN_SIZE_EXP_RANGE[0], MNIST_HIDDEN_SIZE_EXP_RANGE[1])]
]
ARCH = []
DATAFORMAT = []
MATH_FIDELITY = []


# Model definition
class MNISTLinear(nn.Module):

    def __init__(
        self, 
        input_size=MNIST_INPUT_FEATURE_SIZE, 
        output_size=MNIST_OUTPUT_FEATURE_SIZE, 
        hidden_size=MNIIST_HIDDEN_SIZE
    ):

        super(MNISTLinear, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return nn.functional.softmax(x)



# @TODO - For now, we are skipping these parameters, because we are not supporting them
# @pytest.mark.parametrize("math_fidelity", MATH_FIDELITY, ids=[f"math_fidelity={item}" for item in MATH_FIDELITY])
# @pytest.mark.parametrize("dataformat", DATAFORMAT, ids=[f"dataformat={item}" for item in DATAFORMAT])
# @pytest.mark.parametrize("arch", ARCH, ids=[f"arch={item}" for item in ARCH])
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE, ids=[f"hidden_size={item}" for item in HIDDEN_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
def test_mnist_linear(
    training,
    batch_size,
    input_size,
    hidden_size,
    # arch,
    # dataformat,
    # math_fidelity,
):

    if training:
        pytest.skip("Training not supported")

    if batch_size > 1:
        pytest.skip("Batch size greater than 1 not supported")

    inputs = [torch.rand(batch_size, input_size)]

    framework_model = MNISTLinear(input_size=input_size, hidden_size=hidden_size)
    fw_out = framework_model(*inputs)

    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    start = time.time()
    co_out = compiled_model(*inputs)
    end = time.time()

    co_out = [co.to("cpu") for co in co_out]
    assert [compare_with_golden_pcc(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)]

    short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    date = subprocess.check_output(['git', 'show', '-s', '--format=%cd', "--date=format:%y-%m-%d", 'HEAD']).decode('ascii').strip()
    machine_name = socket.gethostname()
    total_time = end - start

    samples_per_sec = batch_size / total_time
    model_name = "MNIST Linear"

    print("====================================================================")
    print("| MNIST Benchmark Results:                                         |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {model_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: : {total_time}")
    print(f"| Total samples: {batch_size}")
    print(f"| Sample per second: {samples_per_sec}")
    print(f"| Batch size: {batch_size}")
    print(f"| Input size: {input_size}")
    print(f"| Hidden size: {hidden_size}")
    print("====================================================================")

    # Create a dictionary to store the results and the configuration
    result = {
        "model": model_name,
        "config": "",
        "date": date,
        "hash": short_hash,
        "machine_name": machine_name,
        "samples_per_sec": samples_per_sec,
        "total_samples": batch_size,
        "total_time": total_time,
        "training": training,
        "batch_size": batch_size,
        "output": "",
        "arch": "",
        "chips": "",
        # "dataformat": dataformat,
        # "math_fidelity": math_fidelity,
        "device": "",
        "galaxy": "",
        "perf_analysis": "",
        "load_tti": "",
        "save_tti": "",
        "task": "",
        "evaluation_score": "",
    }

    return result


def mnist_linear_benchmark(config: dict):

    training = config['training']
    batch_size = config['batch_size']
    output_file = config['output']

    input_size = MNIST_INPUT_FEATURE_SIZE if config['input_size'] is None else config['input_size']
    hidden_size = MNIIST_HIDDEN_SIZE if config['hidden_size'] is None else config['hidden_size']

    result = test_mnist_linear(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
    )

    if not output_file:
        output_file = f"forge-benchmark-e2e-mnist_{batch_size}_{input_size}_{hidden_size}.json"

    result["output"] = output_file 

    # Save the results to a file
    with open(output_file, "w") as f:
        json.dump(result, f)
