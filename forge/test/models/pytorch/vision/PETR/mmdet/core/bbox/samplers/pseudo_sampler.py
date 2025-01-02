# Copyright (c) OpenMMLab. All rights reserved.

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass
