# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor

from .single_roiaware_extractor import Single3DRoIAwareExtractor

__all__ = ["SingleRoIExtractor", "Single3DRoIAwareExtractor"]
