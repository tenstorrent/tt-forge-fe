# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""
