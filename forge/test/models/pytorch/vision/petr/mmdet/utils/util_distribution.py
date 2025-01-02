# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
import torch


def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    is_device_available = {"cuda": torch.cuda.is_available(), "mlu": is_mlu_available()}
    device_list = [k for k, v in is_device_available.items() if v]
    return device_list[0] if len(device_list) == 1 else "cpu"
