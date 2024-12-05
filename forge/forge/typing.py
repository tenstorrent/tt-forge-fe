# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Third Party
import tensorflow as tf
import torch

from .module import ForgeModule
from .tensor import Tensor

FrameworkModule = torch.nn.Module | tf.keras.Model
FrameworkTensor = torch.Tensor | tf.Tensor
AnyModule = FrameworkModule | ForgeModule
AnyTensor = FrameworkTensor | Tensor
