# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from mmcv.utils import Registry

FUSERS = Registry("fusers")


def build_fuser(cfg):
    return FUSERS.build(cfg)
