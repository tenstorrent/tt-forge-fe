# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from .mnist_linear import mnist_linear_benchmark
from .resnet_hf import resnet_hf_benchmark
from .resnet_hf_config import resnet_hf_config_benchmark
from .llama import llama_prefill_benchmark
from .mobilenetv2_basic import mobilenetv2_basic_benchmark
from .efficientnet_timm import efficientnet_timm_benchmark
from .segformer import segformer_classification_benchmark
from .vit import vit_base_benchmark
from .vovnet import vovnet_osmr_benchmark
