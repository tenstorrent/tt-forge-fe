# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from .mnist_linear import mnist_linear_benchmark
from .resnet_hf import resnet_hf_benchmark
from .resnet_hf_config import resnet_hf_config_benchmark
from .llama import llama_prefill_benchmark
from .mobilenetv2_basic import mobilenetv2_basic_benchmark
from .efficientnet_timm import efficientnet_timm_benchmark
from .segformer import segformer_benchmark
from .vit import vit_base_benchmark
from .vovnet import vovnet_osmr_benchmark
from .yolo_v8 import yolo_v8_benchmark
from .yolo_v9 import yolo_v9_benchmark
from .yolo_v4 import yolo_v4_benchmark
from .yolo_v10 import yolo_v10_benchmark
from .unet import unet_benchmark
