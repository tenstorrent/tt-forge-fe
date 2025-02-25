# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

TRANSFORMER = Registry("Transformer")
LINEAR_LAYERS = Registry("linear layers")


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


LINEAR_LAYERS.register_module("Linear", module=nn.Linear)
