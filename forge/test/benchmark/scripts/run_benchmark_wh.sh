# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# Models we run on Wormhole B0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #


# MNIST Linear
python forge/test/benchmark/benchmark.py -m mnist_linear -bs 1 -lp 32 -o forge-benchmark-e2e-mnist.json

# Resnet HF
PYTHONPATH=./forge python forge/test/benchmark/benchmark.py -m resnet50_hf -bs 1 -lp 32 -o forge-benchmark-e2e-resnet50_hf.json
