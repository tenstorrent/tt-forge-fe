# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
import os

import argparse
import torch

# Get the absolute path of the project root and add it to the path
# When we run the tests from benchmark directory it can't find test.utils module,
# so we add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from benchmark import models
from test.utils import reset_seeds

MODELS = {
    "mnist_linear": models.mnist_linear.mnist_linear_benchmark,
    "resnet50_hf": models.resnet_hf.resnet_hf_benchmark,
    "llama": models.llama.llama_prefill_benchmark,
    "mobilenetv2_basic": models.mobilenetv2_basic.mobilenetv2_basic_benchmark,
    "efficientnet_timm": models.efficientnet_timm.efficientnet_timm_benchmark,
}


def read_args():
    """
    Read the arguments from the command line.

    Parameters:
    ----------
    None

    Returns:
    -------
    parsed_args: dict
        The parsed arguments from the command line.
    """

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Benchmark a model on TT hardware")
    parser.add_argument("-m", "--model", help="Model to benchmark (i.e. bert, mnist_linear).")
    parser.add_argument(
        "-c", "--config", default=None, help="Model configuration to benchmark (i.e. tiny, base, large)."
    )
    parser.add_argument("-t", "--training", action="store_true", default=False, help="Benchmark training.")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=1, help="Batch size, number of samples to process at once."
    )
    parser.add_argument("-lp", "--loop_count", type=int, default=1, help="Number of times to run the benchmark.")
    parser.add_argument(
        "-isz",
        "--input_size",
        type=int,
        default=None,
        help="Input size, size of the input sample. If the model gives opportunity to change input size.",
    )
    parser.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size, size of the hidden layer. `If the model gives opportunity to change hidden size.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output json file to write results to, optionally. If file already exists, results will be appended.",
    )

    args = parser.parse_args()

    # Initialize the parsed arguments
    parsed_args = {}

    if not args.model:
        print("\nModel must be specified.\n\n")
        print(parser.print_help())
        exit(1)

    if not args.model in MODELS:
        print("Invalid model name. Available models: ")
        print(list(MODELS.keys()))
        exit(1)

    parsed_args["model"] = args.model
    parsed_args["config"] = args.config
    parsed_args["training"] = args.training
    parsed_args["loop_count"] = args.loop_count

    if not args.batch_size:
        print("\nBatch size is not specified. We set on size 1. \n\n")
        parsed_args["batch_size"] = 1
    else:
        parsed_args["batch_size"] = args.batch_size

    parsed_args["input_size"] = args.input_size
    parsed_args["hidden_size"] = args.hidden_size
    parsed_args["output"] = args.output

    return parsed_args


def run_benchmark(config: dict):
    """
    Run the benchmark test for the given model naconfiguration.

    Parameters:
    ----------
    config: dict
        The configuration of the model.

    Returns:
    -------
    None
    """

    reset_seeds()

    model_func = MODELS[config["model"]]
    model_func(config)


def main():
    """
    Main function for running the benchmark tests.

    Parameters:
    ----------
    None

    Returns:
    -------
    None
    """

    print("Read the arguments from the command line.")
    config = read_args()

    print("Run the benchmark test for the given model configuration.")
    run_benchmark(config)

    print("Done.")


if __name__ == "__main__":
    main()
