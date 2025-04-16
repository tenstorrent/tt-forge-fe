# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# Models we run on Wormhole B0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #


# MNIST Linear
python forge/test/benchmark/benchmark.py -m mnist_linear -bs 32 -lp 32 -o forge-benchmark-e2e-mnist.json

# Resnet HF
python forge/test/benchmark/benchmark.py -m resnet50_hf -bs 8 -lp 32 -o forge-benchmark-e2e-resnet50_hf.json

# Llama
python forge/test/benchmark/benchmark.py -m llama -bs 1 -lp 32 -o forge-benchmark-e2e-llama.json

# MobileNetV2 Basic
python forge/test/benchmark/benchmark.py -m mobilenetv2_basic -bs 1 -lp 32 -o forge-benchmark-e2e-mobilenetv2_basic.json

# EfficientNet Timm
python forge/test/benchmark/benchmark.py -m efficientnet_timm -bs 1 -lp 32 -o forge-benchmark-e2e-efficientnet_timm.json
