# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import tensorflow as tf

from .module import ForgeModule
from .tensor import Tensor

FrameworkModule = torch.nn.Module | tf.keras.Model
FrameworkTensor = torch.Tensor | tf.Tensor
AnyModule = FrameworkModule | ForgeModule
AnyTensor = FrameworkTensor | Tensor
