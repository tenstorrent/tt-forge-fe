# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# Models we run on Wormhole B0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #


# MNIST Linear
python forge/test/benchmark/benchmark.py -m mnist_linear -bs 1 -o forge-benchmark-e2e-mnist.json