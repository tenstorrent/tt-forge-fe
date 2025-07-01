# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# Models we run on Wormhole B0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #


# ==================================================================== #
# Benchmark Arguments
# ==================================================================== #
# -m:  Model name
# -ts: Task type, for example, classification
# -bs: Batch size
# -df: Data format, for example, bfloat16
# -lp: Loop count, number of times to run the model
# -o:  Output file name
# ==================================================================== #

# MNIST Linear
python forge/test/benchmark/benchmark.py -m mnist_linear -bs 32 -lp 32 -o forge-benchmark-e2e-mnist.json

# Resnet HF
python forge/test/benchmark/benchmark.py -m resnet50_hf -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-resnet50_hf.json

# Resnet HF
python forge/test/benchmark/benchmark.py -m resnet50_hf_config -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-resnet50_hf_config.json

# Llama
python forge/test/benchmark/benchmark.py -m llama -bs 1 -df float32 -lp 32 -o forge-benchmark-e2e-llama.json

# MobileNetV2 Basic
python forge/test/benchmark/benchmark.py -m mobilenetv2_basic -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-mobilenetv2_basic.json

# EfficientNet Timm
python forge/test/benchmark/benchmark.py -m efficientnet_timm -ts classification -bs 6 -df bfloat16 -lp 32 -o forge-benchmark-e2e-efficientnet_timm.json

# Segformer Classification
python forge/test/benchmark/benchmark.py -m segformer -bs 1 -df float32 -lp 32 -o forge-benchmark-e2e-segformer.json

# ViT Base
python forge/test/benchmark/benchmark.py -m vit_base -ts classification -bs 1 -df float32 -lp 32 -o forge-benchmark-e2e-vit_base.json

# Vovnet OSMR
python forge/test/benchmark/benchmark.py -m vovnet_timm -ts classification -bs 16 -df bfloat16 -lp 32 -o forge-benchmark-e2e-vovnet_osmr.json

# Yolo4
python forge/test/benchmark/benchmark.py -m yolo_v4 -ts na -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-yolo_v4.json

# Yolo8
python forge/test/benchmark/benchmark.py -m yolo_v8 -ts na -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-yolo_v8.json

# Yolo9
python forge/test/benchmark/benchmark.py -m yolo_v9 -ts na -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-yolo_v9.json

# Yolo10
python forge/test/benchmark/benchmark.py -m yolo_v10 -ts na -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-yolo_v10.json

# Unet
python forge/test/benchmark/benchmark.py -m unet -ts na -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-unet.json
