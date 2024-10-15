# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse

from benchmark import models


MODELS = {
    "mnist_linear": models.mnist_linear.mnist_linear_benchmark,
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

    if not args.batch_size:
        print("\nBatch size is not specified. We set on size 1. \n\n")
        parsed_args["batch_size"] = 1
    else:
        parsed_args["batch_size"] = args.batch_size

    parsed_args["input_size"] = args.input_size
    parsed_args["hidden_size"] = args.hidden_size

    if not args.output:
        print("\nOutput file is not specified.\n\n")
        print(parser.print_help())
        exit(1)

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
