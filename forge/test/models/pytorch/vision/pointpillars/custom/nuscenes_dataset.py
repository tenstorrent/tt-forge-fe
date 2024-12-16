from custom.registry import DATASETS, TRANSFORMS
from custom.imports import Config
from os import path as osp
from typing import Callable, List, Union
import inspect
import numpy as np
import os
from os import path as osp

import numpy as np
import torch
import copy
import functools
import gc
import logging
import pickle
from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Iterator, Generator, Type, Dict, Iterable, Set, TypeVar
from copy import deepcopy
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor
from abc import abstractmethod, ABCMeta
from enum import IntEnum, unique
from pathlib import Path
from contextlib import contextmanager

# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod, abstractproperty, abstractstaticmethod
# from typing import List, Optional, Sequence, Tuple, Type, TypeVar, Union

# import numpy as np
# import torch
from torch import BoolTensor, Tensor

import itertools
from typing import Sequence, Type, TypeVar

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import shapely.geometry as geometry

T = TypeVar('T')


class BaseInstanceMasks(metaclass=ABCMeta):
    """Base class for instance masks."""

    @abstractmethod
    def rescale(self, scale, interpolation='nearest'):
        """Rescale masks as large as possible while keeping the aspect ratio.
        For details can refer to `mmcv.imrescale`.

        Args:
            scale (tuple[int]): The maximum size (h, w) of rescaled mask.
            interpolation (str): Same as :func:`mmcv.imrescale`.

        Returns:
            BaseInstanceMasks: The rescaled masks.
        """

    @abstractmethod
    def resize(self, out_shape, interpolation='nearest'):
        """Resize masks to the given out_shape.

        Args:
            out_shape: Target (h, w) of resized mask.
            interpolation (str): See :func:`mmcv.imresize`.

        Returns:
            BaseInstanceMasks: The resized masks.
        """

    @abstractmethod
    def flip(self, flip_direction='horizontal'):
        """Flip masks alone the given direction.

        Args:
            flip_direction (str): Either 'horizontal' or 'vertical'.

        Returns:
            BaseInstanceMasks: The flipped masks.
        """

    @abstractmethod
    def pad(self, out_shape, pad_val):
        """Pad masks to the given size of (h, w).

        Args:
            out_shape (tuple[int]): Target (h, w) of padded mask.
            pad_val (int): The padded value.

        Returns:
            BaseInstanceMasks: The padded masks.
        """

    @abstractmethod
    def crop(self, bbox):
        """Crop each mask by the given bbox.

        Args:
            bbox (ndarray): Bbox in format [x1, y1, x2, y2], shape (4, ).

        Return:
            BaseInstanceMasks: The cropped masks.
        """

    @abstractmethod
    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        device,
                        interpolation='bilinear',
                        binarize=True):
        """Crop and resize masks by the given bboxes.

        This function is mainly used in mask targets computation.
        It firstly align mask to bboxes by assigned_inds, then crop mask by the
        assigned bbox and resize to the size of (mask_h, mask_w)

        Args:
            bboxes (Tensor): Bboxes in format [x1, y1, x2, y2], shape (N, 4)
            out_shape (tuple[int]): Target (h, w) of resized mask
            inds (ndarray): Indexes to assign masks to each bbox,
                shape (N,) and values should be between [0, num_masks - 1].
            device (str): Device of bboxes
            interpolation (str): See `mmcv.imresize`
            binarize (bool): if True fractional values are rounded to 0 or 1
                after the resize operation. if False and unsupported an error
                will be raised. Defaults to True.

        Return:
            BaseInstanceMasks: the cropped and resized masks.
        """

    @abstractmethod
    def expand(self, expanded_h, expanded_w, top, left):
        """see :class:`Expand`."""

    @property
    @abstractmethod
    def areas(self):
        """ndarray: areas of each instance."""

    @abstractmethod
    def to_ndarray(self):
        """Convert masks to the format of ndarray.

        Return:
            ndarray: Converted masks in the format of ndarray.
        """

    @abstractmethod
    def to_tensor(self, dtype, device):
        """Convert masks to the format of Tensor.

        Args:
            dtype (str): Dtype of converted mask.
            device (torch.device): Device of converted masks.

        Returns:
            Tensor: Converted masks in the format of Tensor.
        """

    @abstractmethod
    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  border_value=0,
                  interpolation='bilinear'):
        """Translate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            border_value (int | float): Border value. Default 0.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            Translated masks.
        """

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        """Shear the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border. Default 0.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            ndarray: Sheared masks.
        """

    @abstractmethod
    def rotate(self, out_shape, angle, center=None, scale=1.0, border_value=0):
        """Rotate the masks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            border_value (int | float): Border value. Default 0 for masks.

        Returns:
            Rotated masks.
        """

    def get_bboxes(self, dst_type='hbb'):
        """Get the certain type boxes from masks.

        Please refer to ``mmdet.structures.bbox.box_type`` for more details of
        the box type.

        Args:
            dst_type: Destination box type.

        Returns:
            :obj:`BaseBoxes`: Certain type boxes.
        """
        from ..bbox import get_box_type
        _, box_type_cls = get_box_type(dst_type)
        return box_type_cls.from_instance_masks(self)

    @classmethod
    @abstractmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[T]): A sequence of mask instances.

        Returns:
            T: Concatenated mask instance.
        """


class BitmapMasks(BaseInstanceMasks):
    """This class represents masks in the form of bitmaps.

    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.data_elements.mask.structures import *  # NOQA
        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(np.int64)
        >>> self = BitmapMasks(masks, height=H, width=W)

        >>> # demo crop_and_resize
        >>> num_boxes = 5
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (14, 14)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> interpolation = 'bilinear'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                assert isinstance(masks[0], np.ndarray)
                assert masks[0].ndim == 2  # (H, W)
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation='nearest'):
        """See :func:`BaseInstanceMasks.rescale`."""
        if len(self.masks) == 0:
            new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack([
                mmcv.imrescale(mask, scale, interpolation=interpolation)
                for mask in self.masks
            ])
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)

    def resize(self, out_shape, interpolation='nearest'):
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack([
                mmcv.imresize(
                    mask, out_shape[::-1], interpolation=interpolation)
                for mask in self.masks
            ])
        return BitmapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical', 'diagonal')

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack([
                mmcv.imflip(mask, direction=flip_direction)
                for mask in self.masks
            ])
        return BitmapMasks(flipped_masks, self.height, self.width)

    def pad(self, out_shape, pad_val=0):
        """See :func:`BaseInstanceMasks.pad`."""
        if len(self.masks) == 0:
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            padded_masks = np.stack([
                mmcv.impad(mask, shape=out_shape, pad_val=pad_val)
                for mask in self.masks
            ])
        return BitmapMasks(padded_masks, *out_shape)

    def crop(self, bbox):
        """See :func:`BaseInstanceMasks.crop`."""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = self.masks[:, y1:y1 + h, x1:x1 + w]
        return BitmapMasks(cropped_masks, h, w)

    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        device='cpu',
                        interpolation='bilinear',
                        binarize=True):
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(
            num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = torch.from_numpy(self.masks).to(device).index_select(
                0, inds).to(dtype=rois.dtype)
            targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape,
                                1.0, 0, 'avg', True).squeeze(1)
            if binarize:
                resized_masks = (targets >= 0.5).cpu().numpy()
            else:
                resized_masks = targets.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    def expand(self, expanded_h, expanded_w, top, left):
        """See :func:`BaseInstanceMasks.expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w),
                                     dtype=np.uint8)
        else:
            expanded_mask = np.zeros((len(self), expanded_h, expanded_w),
                                     dtype=np.uint8)
            expanded_mask[:, top:top + self.height,
                          left:left + self.width] = self.masks
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)

    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  border_value=0,
                  interpolation='bilinear'):
        """Translate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            border_value (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            BitmapMasks: Translated BitmapMasks.

        Example:
            >>> from mmdet.data_elements.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random(dtype=np.uint8)
            >>> out_shape = (32, 32)
            >>> offset = 4
            >>> direction = 'horizontal'
            >>> border_value = 0
            >>> interpolation = 'bilinear'
            >>> # Note, There seem to be issues when:
            >>> # * the mask dtype is not supported by cv2.AffineWarp
            >>> new = self.translate(out_shape, offset, direction,
            >>>                      border_value, interpolation)
            >>> assert len(new) == len(self)
            >>> assert new.height, new.width == out_shape
        """
        if len(self.masks) == 0:
            translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            masks = self.masks
            if masks.shape[-2:] != out_shape:
                empty_masks = np.zeros((masks.shape[0], *out_shape),
                                       dtype=masks.dtype)
                min_h = min(out_shape[0], masks.shape[1])
                min_w = min(out_shape[1], masks.shape[2])
                empty_masks[:, :min_h, :min_w] = masks[:, :min_h, :min_w]
                masks = empty_masks
            translated_masks = mmcv.imtranslate(
                masks.transpose((1, 2, 0)),
                offset,
                direction,
                border_value=border_value,
                interpolation=interpolation)
            if translated_masks.ndim == 2:
                translated_masks = translated_masks[:, :, None]
            translated_masks = translated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(translated_masks, *out_shape)

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        """Shear the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            BitmapMasks: The sheared masks.
        """
        if len(self.masks) == 0:
            sheared_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            sheared_masks = mmcv.imshear(
                self.masks.transpose((1, 2, 0)),
                magnitude,
                direction,
                border_value=border_value,
                interpolation=interpolation)
            if sheared_masks.ndim == 2:
                sheared_masks = sheared_masks[:, :, None]
            sheared_masks = sheared_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(sheared_masks, *out_shape)

    def rotate(self,
               out_shape,
               angle,
               center=None,
               scale=1.0,
               border_value=0,
               interpolation='bilinear'):
        """Rotate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            border_value (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as in :func:`mmcv.imrotate`.

        Returns:
            BitmapMasks: Rotated BitmapMasks.
        """
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = mmcv.imrotate(
                self.masks.transpose((1, 2, 0)),
                angle,
                center=center,
                scale=scale,
                border_value=border_value,
                interpolation=interpolation)
            if rotated_masks.ndim == 2:
                # case when only one mask, (h, w)
                rotated_masks = rotated_masks[:, :, None]  # (h, w, 1)
            rotated_masks = rotated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(rotated_masks, *out_shape)

    @property
    def areas(self):
        """See :py:attr:`BaseInstanceMasks.areas`."""
        return self.masks.sum((1, 2))

    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        return torch.tensor(self.masks, dtype=dtype, device=device)

    @classmethod
    def random(cls,
               num_masks=3,
               height=32,
               width=32,
               dtype=np.uint8,
               rng=None):
        """Generate random bitmap masks for demo / testing purposes.

        Example:
            >>> from mmdet.data_elements.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random()
            >>> print('self = {}'.format(self))
            self = BitmapMasks(num_masks=3, height=32, width=32)
        """
        from mmdet.utils.util_random import ensure_rng
        rng = ensure_rng(rng)
        masks = (rng.rand(num_masks, height, width) > 0.1).astype(dtype)
        self = cls(masks, height=height, width=width)
        return self

    @classmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[BitmapMasks]): A sequence of mask instances.

        Returns:
            BitmapMasks: Concatenated mask instance.
        """
        assert isinstance(masks, Sequence)
        if len(masks) == 0:
            raise ValueError('masks should not be an empty list.')
        assert all(isinstance(m, cls) for m in masks)

        mask_array = np.concatenate([m.masks for m in masks], axis=0)
        return cls(mask_array, *mask_array.shape[1:])


class PolygonMasks(BaseInstanceMasks):
    """This class represents masks in the form of polygons.

    Polygons is a list of three levels. The first level of the list
    corresponds to objects, the second level to the polys that compose the
    object, the third level to the poly coordinates

    Args:
        masks (list[list[ndarray]]): The first level of the list
            corresponds to objects, the second level to the polys that
            compose the object, the third level to the poly coordinates
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.data_elements.mask.structures import *  # NOQA
        >>> masks = [
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 0, 0]) ]
        >>> ]
        >>> height, width = 16, 16
        >>> self = PolygonMasks(masks, height, width)

        >>> # demo translate
        >>> new = self.translate((16, 16), 4., direction='horizontal')
        >>> assert np.all(new.masks[0][0][1::2] == masks[0][0][1::2])
        >>> assert np.all(new.masks[0][0][0::2] == masks[0][0][0::2] + 4)

        >>> # demo crop_and_resize
        >>> num_boxes = 3
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (16, 16)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> interpolation = 'bilinear'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks, height, width):
        assert isinstance(masks, list)
        if len(masks) > 0:
            assert isinstance(masks[0], list)
            assert isinstance(masks[0][0], np.ndarray)

        self.height = height
        self.width = width
        self.masks = masks

    def __getitem__(self, index):
        """Index the polygon masks.

        Args:
            index (ndarray | List): The indices.

        Returns:
            :obj:`PolygonMasks`: The indexed polygon masks.
        """
        if isinstance(index, np.ndarray):
            if index.dtype == bool:
                index = np.where(index)[0].tolist()
            else:
                index = index.tolist()
        if isinstance(index, list):
            masks = [self.masks[i] for i in index]
        else:
            try:
                masks = self.masks[index]
            except Exception:
                raise ValueError(
                    f'Unsupported input of type {type(index)} for indexing!')
        if len(masks) and isinstance(masks[0], np.ndarray):
            masks = [masks]  # ensure a list of three levels
        return PolygonMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)

    def rescale(self, scale, interpolation=None):
        """see :func:`BaseInstanceMasks.rescale`"""
        new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
        if len(self.masks) == 0:
            rescaled_masks = PolygonMasks([], new_h, new_w)
        else:
            rescaled_masks = self.resize((new_h, new_w))
        return rescaled_masks

    def resize(self, out_shape, interpolation=None):
        """see :func:`BaseInstanceMasks.resize`"""
        if len(self.masks) == 0:
            resized_masks = PolygonMasks([], *out_shape)
        else:
            h_scale = out_shape[0] / self.height
            w_scale = out_shape[1] / self.width
            resized_masks = []
            for poly_per_obj in self.masks:
                resized_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] = p[0::2] * w_scale
                    p[1::2] = p[1::2] * h_scale
                    resized_poly.append(p)
                resized_masks.append(resized_poly)
            resized_masks = PolygonMasks(resized_masks, *out_shape)
        return resized_masks

    def flip(self, flip_direction='horizontal'):
        """see :func:`BaseInstanceMasks.flip`"""
        assert flip_direction in ('horizontal', 'vertical', 'diagonal')
        if len(self.masks) == 0:
            flipped_masks = PolygonMasks([], self.height, self.width)
        else:
            flipped_masks = []
            for poly_per_obj in self.masks:
                flipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if flip_direction == 'horizontal':
                        p[0::2] = self.width - p[0::2]
                    elif flip_direction == 'vertical':
                        p[1::2] = self.height - p[1::2]
                    else:
                        p[0::2] = self.width - p[0::2]
                        p[1::2] = self.height - p[1::2]
                    flipped_poly_per_obj.append(p)
                flipped_masks.append(flipped_poly_per_obj)
            flipped_masks = PolygonMasks(flipped_masks, self.height,
                                         self.width)
        return flipped_masks

    def crop(self, bbox):
        """see :func:`BaseInstanceMasks.crop`"""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1

        # clip the boundary
        bbox = bbox.copy()
        bbox[0::2] = np.clip(bbox[0::2], 0, self.width)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.height)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1, 1)
        h = np.maximum(y2 - y1, 1)

        if len(self.masks) == 0:
            cropped_masks = PolygonMasks([], h, w)
        else:
            # reference: https://github.com/facebookresearch/fvcore/blob/main/fvcore/transforms/transform.py  # noqa
            crop_box = geometry.box(x1, y1, x2, y2).buffer(0.0)
            cropped_masks = []
            # suppress shapely warnings util it incorporates GEOS>=3.11.2
            # reference: https://github.com/shapely/shapely/issues/1345
            initial_settings = np.seterr()
            np.seterr(invalid='ignore')
            for poly_per_obj in self.masks:
                cropped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p = geometry.Polygon(p.reshape(-1, 2)).buffer(0.0)
                    # polygon must be valid to perform intersection.
                    if not p.is_valid:
                        continue
                    cropped = p.intersection(crop_box)
                    if cropped.is_empty:
                        continue
                    if isinstance(cropped,
                                  geometry.collection.BaseMultipartGeometry):
                        cropped = cropped.geoms
                    else:
                        cropped = [cropped]
                    # one polygon may be cropped to multiple ones
                    for poly in cropped:
                        # ignore lines or points
                        if not isinstance(
                                poly, geometry.Polygon) or not poly.is_valid:
                            continue
                        coords = np.asarray(poly.exterior.coords)
                        # remove an extra identical vertex at the end
                        coords = coords[:-1]
                        coords[:, 0] -= x1
                        coords[:, 1] -= y1
                        cropped_poly_per_obj.append(coords.reshape(-1))
                # a dummy polygon to avoid misalignment between masks and boxes
                if len(cropped_poly_per_obj) == 0:
                    cropped_poly_per_obj = [np.array([0, 0, 0, 0, 0, 0])]
                cropped_masks.append(cropped_poly_per_obj)
            np.seterr(**initial_settings)
            cropped_masks = PolygonMasks(cropped_masks, h, w)
        return cropped_masks

    def pad(self, out_shape, pad_val=0):
        """padding has no effect on polygons`"""
        return PolygonMasks(self.masks, *out_shape)

    def expand(self, *args, **kwargs):
        """TODO: Add expand for polygon"""
        raise NotImplementedError

    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        device='cpu',
                        interpolation='bilinear',
                        binarize=True):
        """see :func:`BaseInstanceMasks.crop_and_resize`"""
        out_h, out_w = out_shape
        if len(self.masks) == 0:
            return PolygonMasks([], out_h, out_w)

        if not binarize:
            raise ValueError('Polygons are always binary, '
                             'setting binarize=False is unsupported')

        resized_masks = []
        for i in range(len(bboxes)):
            mask = self.masks[inds[i]]
            bbox = bboxes[i, :]
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1, 1)
            h = np.maximum(y2 - y1, 1)
            h_scale = out_h / max(h, 0.1)  # avoid too large scale
            w_scale = out_w / max(w, 0.1)

            resized_mask = []
            for p in mask:
                p = p.copy()
                # crop
                # pycocotools will clip the boundary
                p[0::2] = p[0::2] - bbox[0]
                p[1::2] = p[1::2] - bbox[1]

                # resize
                p[0::2] = p[0::2] * w_scale
                p[1::2] = p[1::2] * h_scale
                resized_mask.append(p)
            resized_masks.append(resized_mask)
        return PolygonMasks(resized_masks, *out_shape)

    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  border_value=None,
                  interpolation=None):
        """Translate the PolygonMasks.

        Example:
            >>> self = PolygonMasks.random(dtype=np.int64)
            >>> out_shape = (self.height, self.width)
            >>> new = self.translate(out_shape, 4., direction='horizontal')
            >>> assert np.all(new.masks[0][0][1::2] == self.masks[0][0][1::2])
            >>> assert np.all(new.masks[0][0][0::2] == self.masks[0][0][0::2] + 4)  # noqa: E501
        """
        assert border_value is None or border_value == 0, \
            'Here border_value is not '\
            f'used, and defaultly should be None or 0. got {border_value}.'
        if len(self.masks) == 0:
            translated_masks = PolygonMasks([], *out_shape)
        else:
            translated_masks = []
            for poly_per_obj in self.masks:
                translated_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    if direction == 'horizontal':
                        p[0::2] = np.clip(p[0::2] + offset, 0, out_shape[1])
                    elif direction == 'vertical':
                        p[1::2] = np.clip(p[1::2] + offset, 0, out_shape[0])
                    translated_poly_per_obj.append(p)
                translated_masks.append(translated_poly_per_obj)
            translated_masks = PolygonMasks(translated_masks, *out_shape)
        return translated_masks

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        """See :func:`BaseInstanceMasks.shear`."""
        if len(self.masks) == 0:
            sheared_masks = PolygonMasks([], *out_shape)
        else:
            sheared_masks = []
            if direction == 'horizontal':
                shear_matrix = np.stack([[1, magnitude],
                                         [0, 1]]).astype(np.float32)
            elif direction == 'vertical':
                shear_matrix = np.stack([[1, 0], [magnitude,
                                                  1]]).astype(np.float32)
            for poly_per_obj in self.masks:
                sheared_poly = []
                for p in poly_per_obj:
                    p = np.stack([p[0::2], p[1::2]], axis=0)  # [2, n]
                    new_coords = np.matmul(shear_matrix, p)  # [2, n]
                    new_coords[0, :] = np.clip(new_coords[0, :], 0,
                                               out_shape[1])
                    new_coords[1, :] = np.clip(new_coords[1, :], 0,
                                               out_shape[0])
                    sheared_poly.append(
                        new_coords.transpose((1, 0)).reshape(-1))
                sheared_masks.append(sheared_poly)
            sheared_masks = PolygonMasks(sheared_masks, *out_shape)
        return sheared_masks

    def rotate(self,
               out_shape,
               angle,
               center=None,
               scale=1.0,
               border_value=0,
               interpolation='bilinear'):
        """See :func:`BaseInstanceMasks.rotate`."""
        if len(self.masks) == 0:
            rotated_masks = PolygonMasks([], *out_shape)
        else:
            rotated_masks = []
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
            for poly_per_obj in self.masks:
                rotated_poly = []
                for p in poly_per_obj:
                    p = p.copy()
                    coords = np.stack([p[0::2], p[1::2]], axis=1)  # [n, 2]
                    # pad 1 to convert from format [x, y] to homogeneous
                    # coordinates format [x, y, 1]
                    coords = np.concatenate(
                        (coords, np.ones((coords.shape[0], 1), coords.dtype)),
                        axis=1)  # [n, 3]
                    rotated_coords = np.matmul(
                        rotate_matrix[None, :, :],
                        coords[:, :, None])[..., 0]  # [n, 2, 1] -> [n, 2]
                    rotated_coords[:, 0] = np.clip(rotated_coords[:, 0], 0,
                                                   out_shape[1])
                    rotated_coords[:, 1] = np.clip(rotated_coords[:, 1], 0,
                                                   out_shape[0])
                    rotated_poly.append(rotated_coords.reshape(-1))
                rotated_masks.append(rotated_poly)
            rotated_masks = PolygonMasks(rotated_masks, *out_shape)
        return rotated_masks

    def to_bitmap(self):
        """convert polygon masks to bitmap masks."""
        bitmap_masks = self.to_ndarray()
        return BitmapMasks(bitmap_masks, self.height, self.width)

    @property
    def areas(self):
        """Compute areas of masks.

        This func is modified from `detectron2
        <https://github.com/facebookresearch/detectron2/blob/ffff8acc35ea88ad1cb1806ab0f00b4c1c5dbfd9/detectron2/structures/masks.py#L387>`_.
        The function only works with Polygons using the shoelace formula.

        Return:
            ndarray: areas of each instance
        """  # noqa: W501
        area = []
        for polygons_per_obj in self.masks:
            area_per_obj = 0
            for p in polygons_per_obj:
                area_per_obj += self._polygon_area(p[0::2], p[1::2])
            area.append(area_per_obj)
        return np.asarray(area)

    def _polygon_area(self, x, y):
        """Compute the area of a component of a polygon.

        Using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Args:
            x (ndarray): x coordinates of the component
            y (ndarray): y coordinates of the component

        Return:
            float: the are of the component
        """  # noqa: 501
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def to_ndarray(self):
        """Convert masks to the format of ndarray."""
        if len(self.masks) == 0:
            return np.empty((0, self.height, self.width), dtype=np.uint8)
        bitmap_masks = []
        for poly_per_obj in self.masks:
            bitmap_masks.append(
                polygon_to_bitmap(poly_per_obj, self.height, self.width))
        return np.stack(bitmap_masks)

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        if len(self.masks) == 0:
            return torch.empty((0, self.height, self.width),
                               dtype=dtype,
                               device=device)
        ndarray_masks = self.to_ndarray()
        return torch.tensor(ndarray_masks, dtype=dtype, device=device)

    @classmethod
    def random(cls,
               num_masks=3,
               height=32,
               width=32,
               n_verts=5,
               dtype=np.float32,
               rng=None):
        """Generate random polygon masks for demo / testing purposes.

        Adapted from [1]_

        References:
            .. [1] https://gitlab.kitware.com/computer-vision/kwimage/-/blob/928cae35ca8/kwimage/structs/polygon.py#L379  # noqa: E501

        Example:
            >>> from mmdet.data_elements.mask.structures import PolygonMasks
            >>> self = PolygonMasks.random()
            >>> print('self = {}'.format(self))
        """
        from mmdet.utils.util_random import ensure_rng
        rng = ensure_rng(rng)

        def _gen_polygon(n, irregularity, spikeyness):
            """Creates the polygon by sampling points on a circle around the
            centre.  Random noise is added by varying the angular spacing
            between sequential points, and by varying the radial distance of
            each point from the centre.

            Based on original code by Mike Ounsworth

            Args:
                n (int): number of vertices
                irregularity (float): [0,1] indicating how much variance there
                    is in the angular spacing of vertices. [0,1] will map to
                    [0, 2pi/numberOfVerts]
                spikeyness (float): [0,1] indicating how much variance there is
                    in each vertex from the circle of radius aveRadius. [0,1]
                    will map to [0, aveRadius]

            Returns:
                a list of vertices, in CCW order.
            """
            from scipy.stats import truncnorm

            # Generate around the unit circle
            cx, cy = (0.0, 0.0)
            radius = 1

            tau = np.pi * 2

            irregularity = np.clip(irregularity, 0, 1) * 2 * np.pi / n
            spikeyness = np.clip(spikeyness, 1e-9, 1)

            # generate n angle steps
            lower = (tau / n) - irregularity
            upper = (tau / n) + irregularity
            angle_steps = rng.uniform(lower, upper, n)

            # normalize the steps so that point 0 and point n+1 are the same
            k = angle_steps.sum() / (2 * np.pi)
            angles = (angle_steps / k).cumsum() + rng.uniform(0, tau)

            # Convert high and low values to be wrt the standard normal range
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            low = 0
            high = 2 * radius
            mean = radius
            std = spikeyness
            a = (low - mean) / std
            b = (high - mean) / std
            tnorm = truncnorm(a=a, b=b, loc=mean, scale=std)

            # now generate the points
            radii = tnorm.rvs(n, random_state=rng)
            x_pts = cx + radii * np.cos(angles)
            y_pts = cy + radii * np.sin(angles)

            points = np.hstack([x_pts[:, None], y_pts[:, None]])

            # Scale to 0-1 space
            points = points - points.min(axis=0)
            points = points / points.max(axis=0)

            # Randomly place within 0-1 space
            points = points * (rng.rand() * .8 + .2)
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)

            high = (1 - max_pt)
            low = (0 - min_pt)
            offset = (rng.rand(2) * (high - low)) + low
            points = points + offset
            return points

        def _order_vertices(verts):
            """
            References:
                https://stackoverflow.com/questions/1709283/how-can-i-sort-a-coordinate-list-for-a-rectangle-counterclockwise
            """
            mlat = verts.T[0].sum() / len(verts)
            mlng = verts.T[1].sum() / len(verts)

            tau = np.pi * 2
            angle = (np.arctan2(mlat - verts.T[0], verts.T[1] - mlng) +
                     tau) % tau
            sortx = angle.argsort()
            verts = verts.take(sortx, axis=0)
            return verts

        # Generate a random exterior for each requested mask
        masks = []
        for _ in range(num_masks):
            exterior = _order_vertices(_gen_polygon(n_verts, 0.9, 0.9))
            exterior = (exterior * [(width, height)]).astype(dtype)
            masks.append([exterior.ravel()])

        self = cls(masks, height, width)
        return self

    @classmethod
    def cat(cls: Type[T], masks: Sequence[T]) -> T:
        """Concatenate a sequence of masks into one single mask instance.

        Args:
            masks (Sequence[PolygonMasks]): A sequence of mask instances.

        Returns:
            PolygonMasks: Concatenated mask instance.
        """
        assert isinstance(masks, Sequence)
        if len(masks) == 0:
            raise ValueError('masks should not be an empty list.')
        assert all(isinstance(m, cls) for m in masks)

        mask_list = list(itertools.chain(*[m.masks for m in masks]))
        return cls(mask_list, masks[0].height, masks[0].width)


def polygon_to_bitmap(polygons, height, width):
    """Convert masks from the form of polygons to bitmaps.

    Args:
        polygons (list[ndarray]): masks in polygon representation
        height (int): mask height
        width (int): mask width

    Return:
        ndarray: the converted masks in bitmap representation
    """
    rles = maskUtils.frPyObjects(polygons, height, width)
    rle = maskUtils.merge(rles)
    bitmap_mask = maskUtils.decode(rle).astype(bool)
    return bitmap_mask


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole


T = TypeVar('T')
DeviceType = Union[str, torch.device]
IndexType = Union[slice, int, list, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor, np.ndarray]
MaskType = Union[BitmapMasks, PolygonMasks]


class BaseBoxes(metaclass=ABCMeta):
    """The base class for 2D box types.

    The functions of ``BaseBoxes`` lie in three fields:

    - Verify the boxes shape.
    - Support tensor-like operations.
    - Define abstract functions for 2D boxes.

    In ``__init__`` , ``BaseBoxes`` verifies the validity of the data shape
    w.r.t ``box_dim``. The tensor with the dimension >= 2 and the length
    of the last dimension being ``box_dim`` will be regarded as valid.
    ``BaseBoxes`` will restore them at the field ``tensor``. It's necessary
    to override ``box_dim`` in subclass to guarantee the data shape is
    correct.

    There are many basic tensor-like functions implemented in ``BaseBoxes``.
    In most cases, users can operate ``BaseBoxes`` instance like a normal
    tensor. To protect the validity of data shape, All tensor-like functions
    cannot modify the last dimension of ``self.tensor``.

    When creating a new box type, users need to inherit from ``BaseBoxes``
    and override abstract methods and specify the ``box_dim``. Then, register
    the new box type by using the decorator ``register_box_type``.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape
            (..., box_dim).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    # Used to verify the last dimension length
    # Should override it in subclass.
    box_dim: int = 0

    def __init__(self,
                 data: Union[Tensor, np.ndarray, Sequence],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[DeviceType] = None,
                 clone: bool = True) -> None:
        if isinstance(data, (np.ndarray, Tensor, Sequence)):
            data = torch.as_tensor(data)
        else:
            raise TypeError('boxes should be Tensor, ndarray, or Sequence, ',
                            f'but got {type(data)}')

        if device is not None or dtype is not None:
            data = data.to(dtype=dtype, device=device)
        # Clone the data to avoid potential bugs
        if clone:
            data = data.clone()
        # handle the empty input like []
        if data.numel() == 0:
            data = data.reshape((-1, self.box_dim))

        assert data.dim() >= 2 and data.size(-1) == self.box_dim, \
            ('The boxes dimension must >= 2 and the length of the last '
             f'dimension must be {self.box_dim}, but got boxes with '
             f'shape {data.shape}.')
        self.tensor = data

    def convert_to(self, dst_type: Union[str, type]) -> 'BaseBoxes':
        """Convert self to another box type.

        Args:
            dst_type (str or type): destination box type.

        Returns:
            :obj:`BaseBoxes`: destination box type object .
        """
        from .box_type import convert_box_type
        return convert_box_type(self, dst_type=dst_type)

    def empty_boxes(self: T,
                    dtype: Optional[torch.dtype] = None,
                    device: Optional[DeviceType] = None) -> T:
        """Create empty box.

        Args:
            dtype (torch.dtype, Optional): data type of boxes.
            device (str or torch.device, Optional): device of boxes.

        Returns:
            T: empty boxes with shape of (0, box_dim).
        """
        empty_box = self.tensor.new_zeros(
            0, self.box_dim, dtype=dtype, device=device)
        return type(self)(empty_box, clone=False)

    def fake_boxes(self: T,
                   sizes: Tuple[int],
                   fill: float = 0,
                   dtype: Optional[torch.dtype] = None,
                   device: Optional[DeviceType] = None) -> T:
        """Create fake boxes with specific sizes and fill values.

        Args:
            sizes (Tuple[int]): The size of fake boxes. The last value must
                be equal with ``self.box_dim``.
            fill (float): filling value. Defaults to 0.
            dtype (torch.dtype, Optional): data type of boxes.
            device (str or torch.device, Optional): device of boxes.

        Returns:
            T: Fake boxes with shape of ``sizes``.
        """
        fake_boxes = self.tensor.new_full(
            sizes, fill, dtype=dtype, device=device)
        return type(self)(fake_boxes, clone=False)

    def __getitem__(self: T, index: IndexType) -> T:
        """Rewrite getitem to protect the last dimension shape."""
        boxes = self.tensor
        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.device)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < boxes.dim()
        elif isinstance(index, tuple):
            assert len(index) < boxes.dim()
            # `Ellipsis`(...) is commonly used in index like [None, ...].
            # When `Ellipsis` is in index, it must be the last item.
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        boxes = boxes[index]
        if boxes.dim() == 1:
            boxes = boxes.reshape(1, -1)
        return type(self)(boxes, clone=False)

    def __setitem__(self: T, index: IndexType, values: Union[Tensor, T]) -> T:
        """Rewrite setitem to protect the last dimension shape."""
        assert type(values) is type(self), \
            'The value to be set must be the same box type as self'
        values = values.tensor

        if isinstance(index, np.ndarray):
            index = torch.as_tensor(index, device=self.device)
        if isinstance(index, Tensor) and index.dtype == torch.bool:
            assert index.dim() < self.tensor.dim()
        elif isinstance(index, tuple):
            assert len(index) < self.tensor.dim()
            # `Ellipsis`(...) is commonly used in index like [None, ...].
            # When `Ellipsis` is in index, it must be the last item.
            if Ellipsis in index:
                assert index[-1] is Ellipsis

        self.tensor[index] = values

    def __len__(self) -> int:
        """Return the length of self.tensor first dimension."""
        return self.tensor.size(0)

    def __deepcopy__(self, memo):
        """Only clone the ``self.tensor`` when applying deepcopy."""
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other
        other.tensor = self.tensor.clone()
        return other

    def __repr__(self) -> str:
        """Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n' + str(self.tensor) + ')'

    def new_tensor(self, *args, **kwargs) -> Tensor:
        """Reload ``new_tensor`` from self.tensor."""
        return self.tensor.new_tensor(*args, **kwargs)

    def new_full(self, *args, **kwargs) -> Tensor:
        """Reload ``new_full`` from self.tensor."""
        return self.tensor.new_full(*args, **kwargs)

    def new_empty(self, *args, **kwargs) -> Tensor:
        """Reload ``new_empty`` from self.tensor."""
        return self.tensor.new_empty(*args, **kwargs)

    def new_ones(self, *args, **kwargs) -> Tensor:
        """Reload ``new_ones`` from self.tensor."""
        return self.tensor.new_ones(*args, **kwargs)

    def new_zeros(self, *args, **kwargs) -> Tensor:
        """Reload ``new_zeros`` from self.tensor."""
        return self.tensor.new_zeros(*args, **kwargs)

    def size(self, dim: Optional[int] = None) -> Union[int, torch.Size]:
        """Reload new_zeros from self.tensor."""
        # self.tensor.size(dim) cannot work when dim=None.
        return self.tensor.size() if dim is None else self.tensor.size(dim)

    def dim(self) -> int:
        """Reload ``dim`` from self.tensor."""
        return self.tensor.dim()

    @property
    def device(self) -> torch.device:
        """Reload ``device`` from self.tensor."""
        return self.tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """Reload ``dtype`` from self.tensor."""
        return self.tensor.dtype

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def numel(self) -> int:
        """Reload ``numel`` from self.tensor."""
        return self.tensor.numel()

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self: T, *args, **kwargs) -> T:
        """Reload ``to`` from self.tensor."""
        return type(self)(self.tensor.to(*args, **kwargs), clone=False)

    def cpu(self: T) -> T:
        """Reload ``cpu`` from self.tensor."""
        return type(self)(self.tensor.cpu(), clone=False)

    def cuda(self: T, *args, **kwargs) -> T:
        """Reload ``cuda`` from self.tensor."""
        return type(self)(self.tensor.cuda(*args, **kwargs), clone=False)

    def clone(self: T) -> T:
        """Reload ``clone`` from self.tensor."""
        return type(self)(self.tensor)

    def detach(self: T) -> T:
        """Reload ``detach`` from self.tensor."""
        return type(self)(self.tensor.detach(), clone=False)

    def view(self: T, *shape: Tuple[int]) -> T:
        """Reload ``view`` from self.tensor."""
        return type(self)(self.tensor.view(shape), clone=False)

    def reshape(self: T, *shape: Tuple[int]) -> T:
        """Reload ``reshape`` from self.tensor."""
        return type(self)(self.tensor.reshape(shape), clone=False)

    def expand(self: T, *sizes: Tuple[int]) -> T:
        """Reload ``expand`` from self.tensor."""
        return type(self)(self.tensor.expand(sizes), clone=False)

    def repeat(self: T, *sizes: Tuple[int]) -> T:
        """Reload ``repeat`` from self.tensor."""
        return type(self)(self.tensor.repeat(sizes), clone=False)

    def transpose(self: T, dim0: int, dim1: int) -> T:
        """Reload ``transpose`` from self.tensor."""
        ndim = self.tensor.dim()
        assert dim0 != -1 and dim0 != ndim - 1
        assert dim1 != -1 and dim1 != ndim - 1
        return type(self)(self.tensor.transpose(dim0, dim1), clone=False)

    def permute(self: T, *dims: Tuple[int]) -> T:
        """Reload ``permute`` from self.tensor."""
        assert dims[-1] == -1 or dims[-1] == self.tensor.dim() - 1
        return type(self)(self.tensor.permute(dims), clone=False)

    def split(self: T,
              split_size_or_sections: Union[int, Sequence[int]],
              dim: int = 0) -> List[T]:
        """Reload ``split`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.split(split_size_or_sections, dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def chunk(self: T, chunks: int, dim: int = 0) -> List[T]:
        """Reload ``chunk`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.chunk(chunks, dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def unbind(self: T, dim: int = 0) -> T:
        """Reload ``unbind`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim() - 1
        boxes_list = self.tensor.unbind(dim=dim)
        return [type(self)(boxes, clone=False) for boxes in boxes_list]

    def flatten(self: T, start_dim: int = 0, end_dim: int = -2) -> T:
        """Reload ``flatten`` from self.tensor."""
        assert end_dim != -1 and end_dim != self.tensor.dim() - 1
        return type(self)(self.tensor.flatten(start_dim, end_dim), clone=False)

    def squeeze(self: T, dim: Optional[int] = None) -> T:
        """Reload ``squeeze`` from self.tensor."""
        boxes = self.tensor.squeeze() if dim is None else \
            self.tensor.squeeze(dim)
        return type(self)(boxes, clone=False)

    def unsqueeze(self: T, dim: int) -> T:
        """Reload ``unsqueeze`` from self.tensor."""
        assert dim != -1 and dim != self.tensor.dim()
        return type(self)(self.tensor.unsqueeze(dim), clone=False)

    @classmethod
    def cat(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        """Cancatenates a box instance list into one single box instance.
        Similar to ``torch.cat``.

        Args:
            box_list (Sequence[T]): A sequence of box instances.
            dim (int): The dimension over which the box are concatenated.
                Defaults to 0.

        Returns:
            T: Concatenated box instance.
        """
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError('box_list should not be a empty list.')

        assert dim != -1 and dim != box_list[0].dim() - 1
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = [boxes.tensor for boxes in box_list]
        return cls(torch.cat(th_box_list, dim=dim), clone=False)

    @classmethod
    def stack(cls: Type[T], box_list: Sequence[T], dim: int = 0) -> T:
        """Concatenates a sequence of tensors along a new dimension. Similar to
        ``torch.stack``.

        Args:
            box_list (Sequence[T]): A sequence of box instances.
            dim (int): Dimension to insert. Defaults to 0.

        Returns:
            T: Concatenated box instance.
        """
        assert isinstance(box_list, Sequence)
        if len(box_list) == 0:
            raise ValueError('box_list should not be a empty list.')

        assert dim != -1 and dim != box_list[0].dim()
        assert all(isinstance(boxes, cls) for boxes in box_list)

        th_box_list = [boxes.tensor for boxes in box_list]
        return cls(torch.stack(th_box_list, dim=dim), clone=False)

    @abstractproperty
    def centers(self) -> Tensor:
        """Return a tensor representing the centers of boxes."""
        pass

    @abstractproperty
    def areas(self) -> Tensor:
        """Return a tensor representing the areas of boxes."""
        pass

    @abstractproperty
    def widths(self) -> Tensor:
        """Return a tensor representing the widths of boxes."""
        pass

    @abstractproperty
    def heights(self) -> Tensor:
        """Return a tensor representing the heights of boxes."""
        pass

    @abstractmethod
    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        """Flip boxes horizontally or vertically in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            direction (str): Flip direction, options are "horizontal",
                "vertical" and "diagonal". Defaults to "horizontal"
        """
        pass

    @abstractmethod
    def translate_(self, distances: Tuple[float, float]) -> None:
        """Translate boxes in-place.

        Args:
            distances (Tuple[float, float]): translate distances. The first
                is horizontal distance and the second is vertical distance.
        """
        pass

    @abstractmethod
    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Clip boxes according to the image shape in-place.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
        """
        pass

    @abstractmethod
    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        """Rotate all boxes in-place.

        Args:
            center (Tuple[float, float]): Rotation origin.
            angle (float): Rotation angle represented in degrees. Positive
                values mean clockwise rotation.
        """
        pass

    @abstractmethod
    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        """Geometric transformat boxes in-place.

        Args:
            homography_matrix (Tensor or np.ndarray]):
                Shape (3, 3) for geometric transformation.
        """
        pass

    @abstractmethod
    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        """Rescale boxes w.r.t. rescale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling boxes.
                The length should be 2.
        """
        pass

    @abstractmethod
    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        """Resize the box width and height w.r.t scale_factor in-place.

        Note:
            Both ``rescale_`` and ``resize_`` will enlarge or shrink boxes
            w.r.t ``scale_facotr``. The difference is that ``resize_`` only
            changes the width and the height of boxes, but ``rescale_`` also
            rescales the box centers simultaneously.

        Args:
            scale_factor (Tuple[float, float]): factors for scaling box
                shapes. The length should be 2.
        """
        pass

    @abstractmethod
    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        """Find boxes inside the image.

        Args:
            img_shape (Tuple[int, int]): A tuple of image height and width.
            all_inside (bool): Whether the boxes are all inside the image or
                part inside the image. Defaults to False.
            allowed_border (int): Boxes that extend beyond the image shape
                boundary by more than ``allowed_border`` are considered
                "outside" Defaults to 0.
        Returns:
            BoolTensor: A BoolTensor indicating whether the box is inside
            the image. Assuming the original boxes have shape (m, n, box_dim),
            the output has shape (m, n).
        """
        pass

    @abstractmethod
    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False) -> BoolTensor:
        """Find inside box points. Boxes dimension must be 2.

        Args:
            points (Tensor): Points coordinates. Has shape of (m, 2).
            is_aligned (bool): Whether ``points`` has been aligned with boxes
                or not. If True, the length of boxes and ``points`` should be
                the same. Defaults to False.

        Returns:
            BoolTensor: A BoolTensor indicating whether a point is inside
            boxes. Assuming the boxes has shape of (n, box_dim), if
            ``is_aligned`` is False. The index has shape of (m, n). If
            ``is_aligned`` is True, m should be equal to n and the index has
            shape of (m, ).
        """
        pass

    @abstractstaticmethod
    def overlaps(boxes1: 'BaseBoxes',
                 boxes2: 'BaseBoxes',
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        """Calculate overlap between two set of boxes with their types
        converted to the present box type.

        Args:
            boxes1 (:obj:`BaseBoxes`): BaseBoxes with shape of (m, box_dim)
                or empty.
            boxes2 (:obj:`BaseBoxes`): BaseBoxes with shape of (n, box_dim)
                or empty.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground). Defaults to "iou".
            is_aligned (bool): If True, then m and n must be equal. Defaults
                to False.
            eps (float): A value added to the denominator for numerical
                stability. Defaults to 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
        """
        pass

    @abstractstaticmethod
    def from_instance_masks(masks: MaskType) -> 'BaseBoxes':
        """Create boxes from instance masks.

        Args:
            masks (:obj:`BitmapMasks` or :obj:`PolygonMasks`): BitmapMasks or
                PolygonMasks instance with length of n.

        Returns:
            :obj:`BaseBoxes`: Converted boxes with shape of (n, box_dim).
        """
        pass
    
BoxType = Union[np.ndarray, Tensor, BaseBoxes]

box_types: dict = {}
_box_type_to_name: dict = {}
box_converters: dict = {}


def _register_box(name: str, box_type: Type, force: bool = False) -> None:
    """Register a box type.

    Args:
        name (str): The name of box type.
        box_type (type): Box mode class to be registered.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.
    """
    assert issubclass(box_type, BaseBoxes)
    name = name.lower()

    if not force and (name in box_types or box_type in _box_type_to_name):
        raise KeyError(f'box type {name} has been registered')
    elif name in box_types:
        _box_type = box_types.pop(name)
        _box_type_to_name.pop(_box_type)
    elif box_type in _box_type_to_name:
        _name = _box_type_to_name.pop(box_type)
        box_types.pop(_name)

    box_types[name] = box_type
    _box_type_to_name[box_type] = name


def register_box(name: str,
                 box_type: Type = None,
                 force: bool = False) -> Union[Type, Callable]:
    """Register a box type.

    A record will be added to ``bbox_types``, whose key is the box type name
    and value is the box type itself. Simultaneously, a reverse dictionary
    ``_box_type_to_name`` will be updated. It can be used as a decorator or
    a normal function.

    Args:
        name (str): The name of box type.
        bbox_type (type, Optional): Box type class to be registered.
            Defaults to None.
        force (bool): Whether to override the existing box type with the same
            name. Defaults to False.

    Examples:
        >>> from mmdet.structures.bbox import register_box
        >>> from mmdet.structures.bbox import BaseBoxes

        >>> # as a decorator
        >>> @register_box('hbox')
        >>> class HorizontalBoxes(BaseBoxes):
        >>>     pass

        >>> # as a normal function
        >>> class RotatedBoxes(BaseBoxes):
        >>>     pass
        >>> register_box('rbox', RotatedBoxes)
    """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method: register_box(name, box_type=BoxCls)
    if box_type is not None:
        _register_box(name=name, box_type=box_type, force=force)
        return box_type

    # use it as a decorator: @register_box(name)
    def _register(cls):
        _register_box(name=name, box_type=cls, force=force)
        return cls

    return _register


def _register_box_converter(src_type: Union[str, type],
                            dst_type: Union[str, type],
                            converter: Callable,
                            force: bool = False) -> None:
    """Register a box converter.

    Args:
        src_type (str or type): source box type name or class.
        dst_type (str or type): destination box type name or class.
        converter (Callable): Convert function.
        force (bool): Whether to override the existing box type with the same
            name. Defaults to False.
    """
    assert callable(converter)
    src_type_name, _ = get_box_type(src_type)
    dst_type_name, _ = get_box_type(dst_type)

    converter_name = src_type_name + '2' + dst_type_name
    if not force and converter_name in box_converters:
        raise KeyError(f'The box converter from {src_type_name} to '
                       f'{dst_type_name} has been registered.')

    box_converters[converter_name] = converter


def register_box_converter(src_type: Union[str, type],
                           dst_type: Union[str, type],
                           converter: Optional[Callable] = None,
                           force: bool = False) -> Callable:
    """Register a box converter.

    A record will be added to ``box_converter``, whose key is
    '{src_type_name}2{dst_type_name}' and value is the convert function.
    It can be used as a decorator or a normal function.

    Args:
        src_type (str or type): source box type name or class.
        dst_type (str or type): destination box type name or class.
        converter (Callable): Convert function. Defaults to None.
        force (bool): Whether to override the existing box type with the same
            name. Defaults to False.

    Examples:
        >>> from mmdet.structures.bbox import register_box_converter
        >>> # as a decorator
        >>> @register_box_converter('hbox', 'rbox')
        >>> def converter_A(boxes):
        >>>     pass

        >>> # as a normal function
        >>> def converter_B(boxes):
        >>>     pass
        >>> register_box_converter('rbox', 'hbox', converter_B)
    """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')

    # use it as a normal method:
    # register_box_converter(src_type, dst_type, converter=Func)
    if converter is not None:
        _register_box_converter(
            src_type=src_type,
            dst_type=dst_type,
            converter=converter,
            force=force)
        return converter

    # use it as a decorator: @register_box_converter(name)
    def _register(func):
        _register_box_converter(
            src_type=src_type, dst_type=dst_type, converter=func, force=force)
        return func

    return _register

def autocast_get_box_type(box_type: Union[str, type]) -> Tuple[str, type]:
    """get both box type name and class.

    Args:
        box_type (str or type): Single box type name or class.

    Returns:
        Tuple[str, type]: A tuple of box type name and class.
    """
    if isinstance(box_type, str):
        type_name = box_type.lower()
        assert type_name in box_types, \
            f"Box type {type_name} hasn't been registered in box_types."
        type_cls = box_types[type_name]
    elif issubclass(box_type, BaseBoxes):
        assert box_type in _box_type_to_name, \
            f"Box type {box_type} hasn't been registered in box_types."
        type_name = _box_type_to_name[box_type]
        type_cls = box_type
    else:
        raise KeyError('box_type must be a str or class inheriting from '
                       f'BaseBoxes, but got {type(box_type)}.')
    return type_name, type_cls

def convert_box_type(boxes: BoxType,
                     *,
                     src_type: Union[str, type] = None,
                     dst_type: Union[str, type] = None) -> BoxType:
    """Convert boxes from source type to destination type.

    If ``boxes`` is a instance of BaseBoxes, the ``src_type`` will be set
    as the type of ``boxes``.

    Args:
        boxes (np.ndarray or Tensor or :obj:`BaseBoxes`): boxes need to
            convert.
        src_type (str or type, Optional): source box type. Defaults to None.
        dst_type (str or type, Optional): destination box type. Defaults to
            None.

    Returns:
        Union[np.ndarray, Tensor, :obj:`BaseBoxes`]: Converted boxes. It's type
        is consistent with the input's type.
    """
    assert dst_type is not None
    dst_type_name, dst_type_cls = get_box_type(dst_type)

    is_box_cls = False
    is_numpy = False
    if isinstance(boxes, BaseBoxes):
        src_type_name, _ = get_box_type(type(boxes))
        is_box_cls = True
    elif isinstance(boxes, (Tensor, np.ndarray)):
        assert src_type is not None
        src_type_name, _ = get_box_type(src_type)
        if isinstance(boxes, np.ndarray):
            is_numpy = True
    else:
        raise TypeError('boxes must be a instance of BaseBoxes, Tensor or '
                        f'ndarray, but get {type(boxes)}.')

    if src_type_name == dst_type_name:
        return boxes

    converter_name = src_type_name + '2' + dst_type_name
    assert converter_name in box_converters, \
        "Convert function hasn't been registered in box_converters."
    converter = box_converters[converter_name]

    if is_box_cls:
        boxes = converter(boxes.tensor)
        return dst_type_cls(boxes)
    elif is_numpy:
        boxes = converter(torch.from_numpy(boxes))
        return boxes.numpy()
    else:
        return converter(boxes)
    
def autocast_box_type(dst_box_type='hbox') -> Callable:
    """A decorator which automatically casts results['gt_bboxes'] to the
    destination box type.

    It commenly used in mmdet.datasets.transforms to make the transforms up-
    compatible with the np.ndarray type of results['gt_bboxes'].

    The speed of processing of np.ndarray and BaseBoxes data are the same:

    - np.ndarray: 0.0509 img/s
    - BaseBoxes: 0.0551 img/s

    Args:
        dst_box_type (str): Destination box type.
    """
    _, box_type_cls = autocast_get_box_type(dst_box_type)

    def decorator(func: Callable) -> Callable:

        def wrapper(self, results: dict, *args, **kwargs) -> dict:
            if ('gt_bboxes' not in results
                    or isinstance(results['gt_bboxes'], BaseBoxes)):
                return func(self, results)
            elif isinstance(results['gt_bboxes'], np.ndarray):
                results['gt_bboxes'] = box_type_cls(
                    results['gt_bboxes'], clone=False)
                if 'mix_results' in results:
                    for res in results['mix_results']:
                        if isinstance(res['gt_bboxes'], np.ndarray):
                            res['gt_bboxes'] = box_type_cls(
                                res['gt_bboxes'], clone=False)

                _results = func(self, results, *args, **kwargs)

                # In some cases, the function will process gt_bboxes in-place
                # Simultaneously convert inputting and outputting gt_bboxes
                # back to np.ndarray
                if isinstance(_results, dict) and 'gt_bboxes' in _results:
                    if isinstance(_results['gt_bboxes'], BaseBoxes):
                        _results['gt_bboxes'] = _results['gt_bboxes'].numpy()
                if isinstance(results['gt_bboxes'], BaseBoxes):
                    results['gt_bboxes'] = results['gt_bboxes'].numpy()
                return _results
            else:
                raise TypeError(
                    "auto_box_type requires results['gt_bboxes'] to "
                    'be BaseBoxes or np.ndarray, but got '
                    f"{type(results['gt_bboxes'])}")

        return wrapper

    return decorator

class cache_randomness:
    def __init__(self, func):

        # Check `func` is to be bound as an instance method
        if not inspect.isfunction(func):
            raise TypeError('Unsupport callable to decorate with'
                            '@cache_randomness.')
        func_args = inspect.getfullargspec(func).args
        if len(func_args) == 0 or func_args[0] != 'self':
            raise TypeError(
                '@cache_randomness should only be used to decorate '
                'instance methods (the first argument is ``self``).')

        functools.update_wrapper(self, func)
        self.func = func
        self.instance_ref = None

    def __set_name__(self, owner, name):
        # Maintain a record of decorated methods in the class
        if not hasattr(owner, '_methods_with_randomness'):
            setattr(owner, '_methods_with_randomness', [])

        # Here `name` equals to `self.__name__`, i.e., the name of the
        # decorated function, due to the invocation of `update_wrapper` in
        # `self.__init__()`
        owner._methods_with_randomness.append(name)

    def __call__(self, *args, **kwargs):
        # Get the transform instance whose method is decorated
        # by cache_randomness
        instance = self.instance_ref()
        name = self.__name__

        # Check the flag ``self._cache_enabled``, which should be
        # set by the contextmanagers like ``cache_random_parameters```
        cache_enabled = getattr(instance, '_cache_enabled', False)

        if cache_enabled:
            # Initialize the cache of the transform instances. The flag
            # ``cache_enabled``` is set by contextmanagers like
            # ``cache_random_params```.
            if not hasattr(instance, '_cache'):
                setattr(instance, '_cache', {})

            if name not in instance._cache:
                instance._cache[name] = self.func(instance, *args, **kwargs)
            # Return the cached value
            return instance._cache[name]
        else:
            # Clear cache
            if hasattr(instance, '_cache'):
                del instance._cache
            # Return function output
            return self.func(instance, *args, **kwargs)

    def __get__(self, obj, cls):
        self.instance_ref = weakref.ref(obj)
        # Return a copy to avoid multiple transform instances sharing
        # one `cache_randomness` instance, which may cause data races
        # in multithreading cases.
        return copy.copy(self)
    
class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self,
                 results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @abstractmethod
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """

@TRANSFORMS.register_module()
class Pack3DDetInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'img']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]

    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global',
                            'axis_align_matrix')
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        from custom.base import Det3DDataSample, InstanceData, PointData
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'images' in results:
                if len(results['images'].keys()) == 1:
                    cam_type = list(results['images'].keys())[0]
                    # single-view image
                    if key in results['images'][cam_type]:
                        data_metas[key] = results['images'][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results['images'].keys())
                    for cam_type in cam_types:
                        if key in results['images'][cam_type]:
                            img_metas.append(results['images'][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif 'lidar_points' in results:
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
    
@TRANSFORMS.register_module()
class PointsRangeFilter(BaseTransform):
    """Filter points by the range.

    Required Keys:

    - points
    - pts_instance_mask (optional)

    Modified Keys:

    - points
    - pts_instance_mask (optional)

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        """Transform function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str
    
@TRANSFORMS.register_module()
class MMCV_RandomFlip(BaseTransform):
    def __init__(self,
                 prob: Optional[Union[float, Iterable[float]]] = None,
                 direction: Union[str, Sequence[Optional[str]]] = 'horizontal',
                 swap_seg_labels: Optional[Sequence] = None) -> None:
        if isinstance(prob, list):
            # assert mmengine.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but \
                              got `{type(prob)}`.')
        self.prob = prob
        self.swap_seg_labels = swap_seg_labels

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmengine.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str, \
                               but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    def _flip_bbox(self, bboxes: np.ndarray, img_shape: Tuple[int, int],
                   direction: str) -> np.ndarray:
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical', and 'diagonal'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        return flipped

    def _flip_keypoints(
        self,
        keypoints: np.ndarray,
        img_shape: Tuple[int, int],
        direction: str,
    ) -> np.ndarray:
        """Flip keypoints horizontally, vertically or diagonally.

        Args:
            keypoints (numpy.ndarray): Keypoints, shape (..., 2)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical', and 'diagonal'.

        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        meta_info = keypoints[..., 2:]
        keypoints = keypoints[..., :2]
        flipped = keypoints.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
        elif direction == 'vertical':
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        elif direction == 'diagonal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        flipped = np.concatenate([flipped, meta_info], axis=-1)
        return flipped

    def _flip_seg_map(self, seg_map: dict, direction: str) -> np.ndarray:
        """Flip segmentation map horizontally, vertically or diagonally.

        Args:
            seg_map (numpy.ndarray): segmentation map, shape (H, W).
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped segmentation map.
        """
        seg_map = mmcv.imflip(seg_map, direction=direction)
        if self.swap_seg_labels is not None:
            # to handle datasets with left/right annotations
            # like 'Left-arm' and 'Right-arm' in LIP dataset
            # Modified from https://github.com/openseg-group/openseg.pytorch/blob/master/lib/datasets/tools/cv2_aug_transforms.py # noqa:E501
            # Licensed under MIT license
            temp = seg_map.copy()
            assert isinstance(self.swap_seg_labels, (tuple, list))
            for pair in self.swap_seg_labels:
                assert isinstance(pair, (tuple, list)) and len(pair) == 2, \
                    'swap_seg_labels must be a sequence with pair, but got ' \
                    f'{self.swap_seg_labels}.'
                seg_map[temp == pair[0]] = pair[1]
                seg_map[temp == pair[1]] = pair[0]
        return seg_map

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      Sequence) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self._flip_keypoints(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip seg map
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = self._flip_seg_map(
                results['gt_seg_map'], direction=results['flip_direction'])
            results['swap_seg_labels'] = self.swap_seg_labels

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints."""
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'flip', and 'flip_direction' keys are
            updated in result dict.
        """
        self._flip_on_direction(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'

        return repr_str
    
@TRANSFORMS.register_module()
class RandomFlip(MMCV_RandomFlip):
    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results['flip_direction']
        h, w = results['img'].shape[:2]

        if cur_dir == 'horizontal':
            homography_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'vertical':
            homography_matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'diagonal':
            homography_matrix = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    # @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)

@TRANSFORMS.register_module()
class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        sync_2d (bool): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float): The flipping probability
            in vertical direction. Defaults to 0.0.
        flip_box3d (bool): Whether to flip bounding box. In most of the case,
            the box should be fliped. In cam-based bev detection, this is set
            to False, since the flip of 2D images does not influence the 3D
            box. Defaults to True.
    """

    def __init__(self,
                 sync_2d: bool = True,
                 flip_ratio_bev_horizontal: float = 0.0,
                 flip_ratio_bev_vertical: float = 0.0,
                 flip_box3d: bool = True,
                 **kwargs) -> None:
        # `flip_ratio_bev_horizontal` is equal to
        # for flip prob of 2d image when
        # `sync_2d` is True
        super(RandomFlip3D, self).__init__(
            prob=flip_ratio_bev_horizontal, direction='horizontal', **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.flip_box3d = flip_box3d
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self,
                            input_dict: dict,
                            direction: str = 'horizontal') -> None:
        """Flip 3D data randomly.

        `random_flip_data_3d` should take these situations into consideration:

        - 1. LIDAR-based 3d detection
        - 2. LIDAR-based 3d segmentation
        - 3. vision-only detection
        - 4. multi-modality 3d detection.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Defaults to 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
            updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if self.flip_box3d:
            if 'gt_bboxes_3d' in input_dict:
                if 'points' in input_dict:
                    input_dict['points'] = input_dict['gt_bboxes_3d'].flip(
                        direction, points=input_dict['points'])
                else:
                    # vision-only detection
                    input_dict['gt_bboxes_3d'].flip(direction)
            else:
                input_dict['points'].flip(direction)

        if 'centers_2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['img_shape'][1]
            input_dict['centers_2d'][..., 0] = \
                w - input_dict['centers_2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints.

        Add the override feature that if 'flip' is already in results, use it
        to do the augmentation.
        """
        if 'flip' not in results:
            cur_dir = self._choose_direction()
        else:
            # `flip_direction` works only when `flip` is True.
            # For example, in `MultiScaleFlipAug3D`, `flip_direction` is
            # 'horizontal' but `flip` is False.
            if results['flip']:
                assert 'flip_direction' in results, 'flip and flip_direction '
                'must exist simultaneously'
                cur_dir = results['flip_direction']
            else:
                cur_dir = None
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, input_dict: dict) -> dict:
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
            'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
            into result dict.
        """
        # flip 2D image and its annotations
        if 'img' in input_dict:
            super(RandomFlip3D, self).transform(input_dict)

        if self.sync_2d and 'img' in input_dict:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio_bev_horizontal else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str
    
@TRANSFORMS.register_module()
class GlobalRotScaleTrans(BaseTransform):
    def __init__(self,
                 rot_range: List[float] = [-0.78539816, 0.78539816],
                 scale_ratio_range: List[float] = [0.95, 1.05],
                 translation_std: List[int] = [0, 0, 0],
                 shift_height: bool = False) -> None:
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'

        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict: dict) -> None:
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        if 'gt_bboxes_3d' in input_dict:
            input_dict['gt_bboxes_3d'].translate(trans_factor)

    def _rot_bbox_points(self, input_dict: dict) -> None:
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            # rotate points with bboxes
            points, rot_mat_T = input_dict['gt_bboxes_3d'].rotate(
                noise_rotation, input_dict['points'])
            input_dict['points'] = points
        else:
            # if no bbox in input_dict, only rotate points
            rot_mat_T = input_dict['points'].rotate(noise_rotation)

        input_dict['pcd_rotation'] = rot_mat_T
        input_dict['pcd_rotation_angle'] = noise_rotation

    def _scale_bbox_points(self, input_dict: dict) -> None:
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points' and
            `gt_bboxes_3d` is updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        if 'gt_bboxes_3d' in input_dict and \
                len(input_dict['gt_bboxes_3d'].tensor) != 0:
            input_dict['gt_bboxes_3d'].scale(scale)

    def _random_scale(self, input_dict: dict) -> None:
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor'
            are updated in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str
    
@TRANSFORMS.register_module()
class MultiScaleFlipAug3D(BaseTransform):
    def __init__(self,
                 transforms: List[dict],
                 img_scale: Optional[Union[Tuple[int], List[Tuple[int]]]],
                 pts_scale_ratio: Union[float, List[float]],
                 flip: bool = False,
                 flip_direction: str = 'horizontal',
                 pcd_horizontal_flip: bool = False,
                 pcd_vertical_flip: bool = False) -> None:
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else [float(pts_scale_ratio)]

        # assert mmengine.is_list_of(self.img_scale, tuple)
        # assert mmengine.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        # assert mmengine.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def transform(self, results: Dict) -> List[Dict]:
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            List[dict]: The list contains the data that is augmented with
            different scales and flips.
        """
        aug_data_list = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            # TODO refactor according to augtest docs
            self.transforms.transforms[0].scale = scale
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data_list.append(data)

        return aug_data_list

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
    
class BasePoints:
    """Base class for Points.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The points
            data with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...).
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self,
                 tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
                 points_dim: int = 3,
                 attribute_dims: Optional[dict] = None) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, points_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, \
            ('The points dimension must be 2 and the length of the last '
             f'dimension must be {points_dim}, but got points with shape '
             f'{tensor.shape}.')

        self.tensor = tensor.clone()
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self) -> Tensor:
        """Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the coordinates of each point.

        Args:
            tensor (Tensor or np.ndarray): Coordinates of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with height of each point in shape
        (N, )."""
        if self.attribute_dims is not None and \
                'height' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['height']]
        else:
            return None

    @height.setter
    def height(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the height of each point.

        Args:
            tensor (Tensor or np.ndarray): Height of each point with shape
                (N, ).
        """
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and \
                'height' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['height']] = tensor
        else:
            # add height attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with color of each point in shape
        (N, 3)."""
        if self.attribute_dims is not None and \
                'color' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['color']]
        else:
            return None

    @color.setter
    def color(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the color of each point.

        Args:
            tensor (Tensor or np.ndarray): Color of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn('point got color value beyond [0, 255]')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and \
                'color' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['color']] = tensor
        else:
            # add color attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(
                dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self) -> torch.Size:
        """torch.Size: Shape of points."""
        return self.tensor.shape

    def shuffle(self) -> Tensor:
        """Shuffle the points.

        Returns:
            Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self,
               rotation: Union[Tensor, np.ndarray, float],
               axis: Optional[int] = None) -> Tensor:
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (Tensor or np.ndarray or float): Rotation matrix or angle.
            axis (int, optional): Axis to rotate at. Defaults to None.

        Returns:
            Tensor: Rotation matrix.
        """
        if not isinstance(rotation, Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, \
            f'invalid rotation shape {rotation.shape}'

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rotated_points, rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, :3][None], rotation, axis=axis, return_mat=True)
            self.tensor[:, :3] = rotated_points.squeeze(0)
            rot_mat_T = rot_mat_T.squeeze(0)
        else:
            # rotation.numel() == 9
            self.tensor[:, :3] = self.tensor[:, :3] @ rotation
            rot_mat_T = rotation

        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction: str = 'horizontal') -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """
        pass

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        """Translate points with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size 3
                or nx3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and \
                trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(
                f'Unsupported translation vector of shape {trans_vector.shape}'
            )
        self.tensor[:, :3] += trans_vector

    def in_range_3d(
            self, point_range: Union[Tensor, np.ndarray,
                                     Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > point_range[0])
                          & (self.tensor[:, 1] > point_range[1])
                          & (self.tensor[:, 2] > point_range[2])
                          & (self.tensor[:, 0] < point_range[3])
                          & (self.tensor[:, 1] < point_range[4])
                          & (self.tensor[:, 2] < point_range[5]))
        return in_range_flags

    @property
    def bev(self) -> Tensor:
        """Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 1]]

    def in_range_bev(
            self, point_range: Union[Tensor, np.ndarray,
                                     Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point in order of (x_min, y_min, x_max, y_max).

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = ((self.bev[:, 0] > point_range[0])
                          & (self.bev[:, 1] > point_range[1])
                          & (self.bev[:, 0] < point_range[2])
                          & (self.bev[:, 1] < point_range[3]))
        return in_range_flags

    @abstractmethod
    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor,
                                          np.ndarray]] = None) -> 'BasePoints':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Point mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type in the
            ``dst`` mode.
        """
        pass

    def scale(self, scale_factor: float) -> None:
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(
            self, item: Union[int, tuple, slice, np.ndarray,
                              Tensor]) -> 'BasePoints':
        """
        Args:
            item (int or tuple or slice or np.ndarray or Tensor): Index of
                points.

        Note:
            The following usage are allowed:

            1. `new_points = points[3]`: Return a `Points` that contains only
               one point.
            2. `new_points = points[2:10]`: Return a slice of points.
            3. `new_points = points[vector]`: Whether vector is a
               torch.BoolTensor with `length = len(points)`. Nonzero elements
               in the vector will be selected.
            4. `new_points = points[3:11, vector]`: Return a slice of points
               and attribute dims.
            5. `new_points = points[4:12, 2]`: Return a slice of points with
               single attribute.

            Note that the returned Points might share storage with this Points,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of :class:`BasePoints` after
            indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                points_dim=self.points_dim,
                attribute_dims=self.attribute_dims)
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] \
                    if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]

            keep_dims = list(
                set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(
                        set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f'Invalid slice {item}!')

        assert p.dim() == 2, \
            f'Indexing on Points with {item} failed to return a matrix!'
        return original_type(
            p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self) -> int:
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, points_list: Sequence['BasePoints']) -> 'BasePoints':
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (Sequence[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        cat_points = cls(
            torch.cat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].points_dim,
            attribute_dims=points_list[0].attribute_dims)
        return cat_points

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args,
           **kwargs) -> 'BasePoints':
        """Convert current points to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new points object on the specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def cpu(self) -> 'BasePoints':
        """Convert current points to cpu device.

        Returns:
            :obj:`BasePoints`: A new points object on the cpu device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cpu(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def cuda(self, *args, **kwargs) -> 'BasePoints':
        """Convert current points to cuda device.

        Returns:
            :obj:`BasePoints`: A new points object on the cuda device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def clone(self) -> 'BasePoints':
        """Clone the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def detach(self) -> 'BasePoints':
        """Detach the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.detach(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    @property
    def device(self) -> torch.device:
        """torch.device: The device of the points are on."""
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        """Yield a point as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A point of shape (points_dim, ).
        """
        yield from self.tensor

    def new_point(
        self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]
    ) -> 'BasePoints':
        """Create a new point object with data.

        The new point and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``, the object's
            other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(
            new_tensor,
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)
    
@TRANSFORMS.register_module()
class LoadPointsFromMultiSweeps(BaseTransform):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points. Defaults to False.
        test_mode (bool): If `test_mode=True`, it will not randomly sample
            sweeps but select the nearest N frames. Defaults to False.
    """

    def __init__(self,
                 sweeps_num: int = 10,
                 load_dim: int = 5,
                 use_dim: List[int] = [0, 1, 2, 4],
                 backend_args: Optional[dict] = None,
                 pad_empty_sweeps: bool = False,
                 remove_close: bool = False,
                 test_mode: bool = False) -> None:
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        self.use_dim = use_dim
        self.backend_args = backend_args
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self,
                      points: Union[np.ndarray, BasePoints],
                      radius: float = 1.0) -> Union[np.ndarray, BasePoints]:
        """Remove point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray | :obj:`BasePoints`: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def transform(self, results: dict) -> dict:
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
            Updated key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                  cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if 'lidar_sweeps' not in results:
            if self.pad_empty_sweeps:
                for i in range(self.sweeps_num):
                    if self.remove_close:
                        sweep_points_list.append(self._remove_close(points))
                    else:
                        sweep_points_list.append(points)
        else:
            if len(results['lidar_sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['lidar_sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['lidar_sweeps']),
                    self.sweeps_num,
                    replace=False)
            for idx in choices:
                sweep = results['lidar_sweeps'][idx]
                points_sweep = self._load_points(
                    sweep['lidar_points']['lidar_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                # bc-breaking: Timestamp has divided 1e6 in pkl infos.
                sweep_ts = sweep['timestamp']
                lidar2sensor = np.array(sweep['lidar_points']['lidar2sensor'])
                points_sweep[:, :
                             3] = points_sweep[:, :3] @ lidar2sensor[:3, :3]
                points_sweep[:, :3] -= lidar2sensor[:3, 3]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'

def get(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bytes:
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.get(filepath)

def get_points_type(points_type: str) -> type:
    """Get the class of points according to coordinate type.

    Args:
        points_type (str): The type of points coordinate. The valid value are
            "CAMERA", "LIDAR" and "DEPTH".

    Returns:
        type: Points type.
    """
    points_type_upper = points_type.upper()
    if points_type_upper == 'CAMERA':
        points_cls = CameraPoints
    elif points_type_upper == 'LIDAR':
        points_cls = LiDARPoints
    elif points_type_upper == 'DEPTH':
        points_cls = DepthPoints
    else:
        raise ValueError('Only "points_type" of "CAMERA", "LIDAR" and "DEPTH" '
                         f'are supported, got {points_type}')

    return points_cls

@TRANSFORMS.register_module()
class LoadPointsFromFile(BaseTransform):
    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            # mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str
    
class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: :meth:`get()` and
    :meth:`get_text()`.

    - :meth:`get()` reads the file as a byte stream.
    - :meth:`get_text()` reads the file as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    # This attribute will be deprecated in future.
    _allow_symlink = False

    @property
    def allow_symlink(self):
        print_log(
            'allow_symlink will be deprecated in future',
            logger='current',
            level=logging.WARNING)
        return self._allow_symlink

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass

class PetrelBackend(BaseStorageBackend):
    """Petrel storage backend (for internal usage).

    PetrelBackend supports reading and writing data to multiple clusters.
    If the file path contains the cluster name, PetrelBackend will read data
    from specified cluster or write data to it. Otherwise, PetrelBackend will
    access the default cluster.

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Petrel path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Defaults to None.
        enable_mc (bool, optional): Whether to enable memcached support.
            Defaults to True.
        conf_path (str, optional): Config path of Petrel client. Default: None.
            `New in version 0.3.3`.

    Examples:
        >>> backend = PetrelBackend()
        >>> filepath1 = 'petrel://path/of/file'
        >>> filepath2 = 'cluster-name:petrel://path/of/file'
        >>> backend.get(filepath1)  # get data from default cluster
        >>> client.get(filepath2)  # get data from 'cluster-name' cluster
    """

    def __init__(self,
                 path_mapping: Optional[dict] = None,
                 enable_mc: bool = True,
                 conf_path: Optional[str] = None):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')

        self._client = client.Client(conf_path=conf_path, enable_mc=enable_mc)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str or Path): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v, 1)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of petrel oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def _replace_prefix(self, filepath: Union[str, Path]) -> str:
        filepath = str(filepath)
        return filepath.replace('petrel://', 's3://')

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Return bytes read from filepath.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        value = self._client.Get(filepath)
        return value

    def get_text(
        self,
        filepath: Union[str, Path],
        encoding: str = 'utf-8',
    ) -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.get_text(filepath)
            'hello world'
        """
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.put(b'hello world', filepath)
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.put(filepath, obj)

    def put_text(
        self,
        obj: str,
        filepath: Union[str, Path],
        encoding: str = 'utf-8',
    ) -> None:
        """Write text to a given ``filepath``.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to encode the ``obj``.
                Defaults to 'utf-8'.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.put_text('hello world', filepath)
        """
        self.put(bytes(obj, encoding=encoding), filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.exists(filepath)
            True
        """
        if not (has_method(self._client, 'contains')
                and has_method(self._client, 'isdir')):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `contains` and `isdir` methods, please use a higher'
                'version or dev branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath) or self._client.isdir(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/dir'
            >>> backend.isdir(filepath)
            True
        """
        if not has_method(self._client, 'isdir'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `isdir` method, please use a higher version or dev'
                ' branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.isfile(filepath)
            True
        """
        if not has_method(self._client, 'contains'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `contains` method, please use a higher version or '
                'dev branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath)

    def join_path(
        self,
        filepath: Union[str, Path],
        *filepaths: Union[str, Path],
    ) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result after concatenation.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.join_path(filepath, 'another/path')
            'petrel://path/of/file/another/path'
            >>> backend.join_path(filepath, '/another/path')
            'petrel://path/of/file/another/path'
        """
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith('/'):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_path = self._format_path(self._map_path(path))
            formatted_paths.append(formatted_path.lstrip('/'))

        return '/'.join(formatted_paths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str or Path): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = PetrelBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> filepath = 'petrel://path/of/file'
            >>> with backend.get_local_path(filepath) as path:
            ...     # do something here
        """
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a file src to dst and return the destination file.

        src and dst should have the same prefix. If dst specifies a directory,
        the file will be copied into dst using the base filename from src. If
        dst specifies a file that already exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> # dst is a file
            >>> src = 'petrel://path/of/file'
            >>> dst = 'petrel://path/of/file1'
            >>> backend.copyfile(src, dst)
            'petrel://path/of/file1'

            >>> # dst is a directory
            >>> dst = 'petrel://path/of/dir'
            >>> backend.copyfile(src, dst)
            'petrel://path/of/dir/file'
        """
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, src.split('/')[-1])

        if src == dst:
            raise SameFileError('src and dst should not be same')

        self.put(self.get(src), dst)
        return dst

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        src and dst should have the same prefix.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'petrel://path/of/dir'
            >>> dst = 'petrel://path/of/dir1'
            >>> backend.copytree(src, dst)
            'petrel://path/of/dir1'
        """
        src = self._format_path(self._map_path(src))
        dst = self._format_path(self._map_path(dst))

        if self.exists(dst):
            raise FileExistsError('dst should not exist')

        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            src_path = self.join_path(src, path)
            dst_path = self.join_path(dst, path)
            self.put(self.get(src_path), dst_path)

        return dst

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Upload a local file src to dst and return the destination file.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = PetrelBackend()
            >>> # dst is a file
            >>> src = 'path/of/your/file'
            >>> dst = 'petrel://path/of/file1'
            >>> backend.copyfile_from_local(src, dst)
            'petrel://path/of/file1'

            >>> # dst is a directory
            >>> dst = 'petrel://path/of/dir'
            >>> backend.copyfile_from_local(src, dst)
            'petrel://path/of/dir/file'
        """
        dst = self._format_path(self._map_path(dst))
        if self.isdir(dst):
            dst = self.join_path(dst, osp.basename(src))

        with open(src, 'rb') as f:
            self.put(f.read(), dst)

        return dst

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'path/of/your/dir'
            >>> dst = 'petrel://path/of/dir1'
            >>> backend.copytree_from_local(src, dst)
            'petrel://path/of/dir1'
        """
        dst = self._format_path(self._map_path(dst))
        if self.exists(dst):
            raise FileExistsError('dst should not exist')

        src = str(src)

        for cur_dir, _, files in os.walk(src):
            for f in files:
                src_path = osp.join(cur_dir, f)
                dst_path = self.join_path(dst, src_path.replace(src, ''))
                self.copyfile_from_local(src_path, dst_path)

        return dst

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        """Copy the file src to local dst and return the destination file.

        If dst specifies a directory, the file will be copied into dst using
        the base filename from src. If dst specifies a file that already
        exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to to local dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = PetrelBackend()
            >>> # dst is a file
            >>> src = 'petrel://path/of/file'
            >>> dst = 'path/of/your/file'
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/file'

            >>> # dst is a directory
            >>> dst = 'path/of/your/dir'
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/dir/file'
        """
        if osp.isdir(dst):
            basename = osp.basename(src)
            if isinstance(dst, str):
                dst = osp.join(dst, basename)
            else:
                assert isinstance(dst, Path)
                dst = dst / basename

        with open(dst, 'wb') as f:
            f.write(self.get(src))

        return dst

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        """Recursively copy an entire directory tree rooted at src to a local
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to local dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'petrel://path/of/dir'
            >>> dst = 'path/of/your/dir'
            >>> backend.copytree_to_local(src, dst)
            'path/of/your/dir'
        """
        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            dst_path = osp.join(dst, path)
            mmengine.mkdir_or_exist(osp.dirname(dst_path))
            with open(dst_path, 'wb') as f:
                f.write(self.get(self.join_path(src, path)))

        return dst

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.remove(filepath)
        """
        if not has_method(self._client, 'delete'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `delete` method, please use a higher version or dev '
                'branch instead.')

        if not self.exists(filepath):
            raise FileNotFoundError(f'filepath {filepath} does not exist')

        if self.isdir(filepath):
            raise IsADirectoryError('filepath should be a file')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        self._client.delete(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> backend = PetrelBackend()
            >>> dir_path = 'petrel://path/of/dir'
            >>> backend.rmtree(dir_path)
        """
        for path in self.list_dir_or_file(
                dir_path, list_dir=False, recursive=True):
            filepath = self.join_path(dir_path, path)
            self.remove(filepath)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        """Create a symbolic link pointing to src named dst.

        Directly copy src to dst because PetrelBacekend does not support create
        a symbolic link.

        Args:
            src (str or Path): A file or directory to be copied.
            dst (str or Path): Copy a file or directory to dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            bool: Return False because PetrelBackend does not support create
            a symbolic link.

        Examples:
            >>> backend = PetrelBackend()
            >>> src = 'petrel://path/of/file'
            >>> dst = 'petrel://path/of/your/file'
            >>> backend.copy_if_symlink_fails(src, dst)
            False
            >>> src = 'petrel://path/of/dir'
            >>> dst = 'petrel://path/of/your/dir'
            >>> backend.copy_if_symlink_fails(src, dst)
            False
        """
        if self.isfile(src):
            self.copyfile(src, dst)
        else:
            self.copytree(src, dst)
        return False

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the
                directory. Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = PetrelBackend()
            >>> dir_path = 'petrel://path/of/dir'
            >>> # list those files and directories in current directory
            >>> for file_path in backend.list_dir_or_file(dir_path):
            ...     print(file_path)
            >>> # only list files
            >>> for file_path in backend.list_dir_or_file(dir_path, list_dir=False):
            ...     print(file_path)
            >>> # only list directories
            >>> for file_path in backend.list_dir_or_file(dir_path, list_file=False):
            ...     print(file_path)
            >>> # only list files ending with specified suffixes
            >>> for file_path in backend.list_dir_or_file(dir_path, suffix='.txt'):
            ...     print(file_path)
            >>> # list all files and directory recursively
            >>> for file_path in backend.list_dir_or_file(dir_path, recursive=True):
            ...     print(file_path)
        """  # noqa: E501
        if not has_method(self._client, 'list'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `list` method, please use a higher version or dev'
                ' branch instead.')

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        dir_path = self._replace_prefix(dir_path)
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith('/'):
            dir_path += '/'

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith('/'):  # a directory path
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root):-1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(next_dir_path, list_dir,
                                                     list_file, suffix,
                                                     recursive)
                else:  # a file path
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root):]
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)

    def generate_presigned_url(self,
                               url: str,
                               client_method: str = 'get_object',
                               expires_in: int = 3600) -> str:
        """Generate the presigned url of video stream which can be passed to
        mmcv.VideoReader. Now only work on Petrel backend.

        Note:
            Now only work on Petrel backend.

        Args:
            url (str): Url of video stream.
            client_method (str): Method of client, 'get_object' or
                'put_object'. Default: 'get_object'.
            expires_in (int): expires, in seconds. Default: 3600.

        Returns:
            str: Generated presigned url.
        """
        return self._client.generate_presigned_url(url, client_method,
                                                   expires_in)


class MemcachedBackend(BaseStorageBackend):
    """Memcached storage backend.

    Attributes:
        server_list_cfg (str): Config file for memcached server list.
        client_cfg (str): Config file for memcached client.
        sys_path (str, optional): Additional path to be appended to `sys.path`.
            Defaults to None.
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError(
                'Please install memcached to enable MemcachedBackend.')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(self.server_list_cfg,
                                                      self.client_cfg)
        # mc.pyvector servers as a point which points to a memory cache
        self._mc_buffer = mc.pyvector()

    def get(self, filepath: Union[str, Path]):
        """Get values according to the filepath.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> server_list_cfg = '/path/of/server_list.conf'
            >>> client_cfg = '/path/of/mc.conf'
            >>> backend = MemcachedBackend(server_list_cfg, client_cfg)
            >>> backend.get('/path/of/file')
            b'hello world'
        """
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError
    
class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
        db_path (str): Lmdb database path.
        readonly (bool): Lmdb environment parameter. If True, disallow any
            write operations. Defaults to True.
        lock (bool): Lmdb environment parameter. If False, when concurrent
            access occurs, do not lock the database. Defaults to False.
        readahead (bool): Lmdb environment parameter. If False, disable the OS
            filesystem readahead mechanism, which may improve random read
            performance when a database is larger than RAM. Defaults to False.
        **kwargs: Keyword arguments passed to `lmdb.open`.

    Attributes:
        db_path (str): Lmdb database path.
    """

    def __init__(self,
                 db_path,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 **kwargs):
        try:
            import lmdb  # noqa: F401
        except ImportError:
            raise ImportError(
                'Please run "pip install lmdb" to enable LmdbBackend.')

        self.db_path = str(db_path)
        self.readonly = readonly
        self.lock = lock
        self.readahead = readahead
        self.kwargs = kwargs
        self._client = None

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Get values according to the filepath.

        Args:
            filepath (str or Path): Here, filepath is the lmdb key.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = LmdbBackend('path/to/lmdb')
            >>> backend.get('key')
            b'hello world'
        """
        if self._client is None:
            self._client = self._get_client()

        filepath = str(filepath)
        with self._client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError

    def _get_client(self):
        import lmdb

        return lmdb.open(
            self.db_path,
            readonly=self.readonly,
            lock=self.lock,
            readahead=self.readahead,
            **self.kwargs)

    def __del__(self):
        if self._client is not None:
            self._client.close()

class LocalBackend(BaseStorageBackend):
    """Raw local storage backend."""

    _allow_symlink = True

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        with open(filepath, 'rb') as f:
            value = f.read()
        return value

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.get_text(filepath)
            'hello world'
        """
        with open(filepath, encoding=encoding) as f:
            text = f.read()
        return text

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.put(b'hello world', filepath)
        """
        mmengine.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, 'wb') as f:
            f.write(obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        """Write text to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.put_text('hello world', filepath)
        """
        mmengine.mkdir_or_exist(osp.dirname(filepath))
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.exists(filepath)
            True
        """
        return osp.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/dir'
            >>> backend.isdir(filepath)
            True
        """
        return osp.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.isfile(filepath)
            True
        """
        return osp.isfile(filepath)

    def join_path(self, filepath: Union[str, Path],
                  *filepaths: Union[str, Path]) -> str:
        r"""Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of \*filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath1 = '/path/of/dir1'
            >>> filepath2 = 'dir2'
            >>> filepath3 = 'path/of/file'
            >>> backend.join_path(filepath1, filepath2, filepath3)
            '/path/of/dir/dir2/path/of/file'
        """
        # TODO, if filepath or filepaths are Path, should return Path
        return osp.join(filepath, *filepaths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Only for unified API and do nothing.

        Args:
            filepath (str or Path): Path to be read data.
            backend_args (dict, optional): Arguments to instantiate the
                corresponding backend. Defaults to None.

        Examples:
            >>> backend = LocalBackend()
            >>> with backend.get_local_path('s3://bucket/abc.jpg') as path:
            ...     # do something here
        """
        yield filepath

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a file src to dst and return the destination file.

        src and dst should have the same prefix. If dst specifies a directory,
        the file will be copied into dst using the base filename from src. If
        dst specifies a file that already exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to '/path1/of/dir/file'
            >>> backend.copyfile(src, dst)
            '/path1/of/dir/file'
        """
        return shutil.copy(src, dst)

    def copytree(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        src and dst should have the same prefix and dst must not already exist.

        TODO: Whether to support dirs_exist_ok parameter.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree(src, dst)
            '/path/of/dir2'
        """
        return shutil.copytree(src, dst)

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a local file src to dst and return the destination file. Same
        as :meth:`copyfile`.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile_from_local(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to
            >>> backend.copyfile_from_local(src, dst)
            '/path1/of/dir/file'
        """
        return self.copyfile(src, dst)

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory. Same as
        :meth:`copytree`.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree_from_local(src, dst)
            '/path/of/dir2'
        """
        return self.copytree(src, dst)

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy the file src to local dst and return the destination file. Same
        as :meth:`copyfile`.

        If dst specifies a directory, the file will be copied into dst using
        the base filename from src. If dst specifies a file that already
        exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to to local dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = LocalBackend()
            >>> # dst is a file
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> # src will be copied to '/path1/of/file1'
            >>> backend.copyfile_to_local(src, dst)
            '/path1/of/file1'

            >>> # dst is a directory
            >>> dst = '/path1/of/dir'
            >>> # src will be copied to
            >>> backend.copyfile_to_local(src, dst)
            '/path1/of/dir/file'
        """
        return self.copyfile(src, dst)

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a local
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A directory to be copied.
            dst (str or Path): Copy directory to local dst.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/dir1'
            >>> dst = '/path/of/dir2'
            >>> backend.copytree_from_local(src, dst)
            '/path/of/dir2'
        """
        return self.copytree(src, dst)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.

        Examples:
            >>> backend = LocalBackend()
            >>> filepath = '/path/of/file'
            >>> backend.remove(filepath)
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f'filepath {filepath} does not exist')

        if self.isdir(filepath):
            raise IsADirectoryError('filepath should be a file')

        os.remove(filepath)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> dir_path = '/path/of/dir'
            >>> backend.rmtree(dir_path)
        """
        shutil.rmtree(dir_path)

    def copy_if_symlink_fails(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> bool:
        """Create a symbolic link pointing to src named dst.

        If failed to create a symbolic link pointing to src, directly copy src
        to dst instead.

        Args:
            src (str or Path): Create a symbolic link pointing to src.
            dst (str or Path): Create a symbolic link named dst.

        Returns:
            bool: Return True if successfully create a symbolic link pointing
            to src. Otherwise, return False.

        Examples:
            >>> backend = LocalBackend()
            >>> src = '/path/of/file'
            >>> dst = '/path1/of/file1'
            >>> backend.copy_if_symlink_fails(src, dst)
            True
            >>> src = '/path/of/dir'
            >>> dst = '/path1/of/dir1'
            >>> backend.copy_if_symlink_fails(src, dst)
            True
        """
        try:
            os.symlink(src, dst)
            return True
        except Exception:
            if self.isfile(src):
                self.copyfile(src, dst)
            else:
                self.copytree(src, dst)
            return False

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

        Args:
            dir_path (str or Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional): File suffix that we are
                interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the directory.
                Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = LocalBackend()
            >>> dir_path = '/path/of/dir'
            >>> # list those files and directories in current directory
            >>> for file_path in backend.list_dir_or_file(dir_path):
            ...     print(file_path)
            >>> # only list files
            >>> for file_path in backend.list_dir_or_file(dir_path, list_dir=False):
            ...     print(file_path)
            >>> # only list directories
            >>> for file_path in backend.list_dir_or_file(dir_path, list_file=False):
            ...     print(file_path)
            >>> # only list files ending with specified suffixes
            >>> for file_path in backend.list_dir_or_file(dir_path, suffix='.txt'):
            ...     print(file_path)
            >>> # list all files and directory recursively
            >>> for file_path in backend.list_dir_or_file(dir_path, recursive=True):
            ...     print(file_path)
        """  # noqa: E501
        if list_dir and suffix is not None:
            raise TypeError('`suffix` should be None when `list_dir` is True')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for entry in os.scandir(dir_path):
                if not entry.name.startswith('.') and entry.is_file():
                    rel_path = osp.relpath(entry.path, root)
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif osp.isdir(entry.path):
                    if list_dir:
                        rel_dir = osp.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(entry.path, list_dir,
                                                     list_file, suffix,
                                                     recursive)

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)
    
class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def get(self, filepath: str) -> bytes:
        """Read bytes from a given ``filepath``.

        Args:
            filepath (str): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get('http://path/of/file')
            b'hello world'
        """
        return urlopen(filepath).read()

    def get_text(self, filepath, encoding='utf-8') -> str:
        """Read text from a given ``filepath``.

        Args:
            filepath (str): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get_text('http://path/of/file')
            'hello world'
        """
        return urlopen(filepath).read().decode(encoding)

    @contextmanager
    def get_local_path(
            self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = HTTPBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with backend.get_local_path('http://path/of/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

backends: dict = {}
prefix_to_backends: dict = {}
backend_instances: dict = {}

def _register_backend(name: str,
                      backend: Type[BaseStorageBackend],
                      force: bool = False,
                      prefixes: Union[str, list, tuple, None] = None):
    """Register a backend.

    Args:
        name (str): The name of the registered backend.
        backend (BaseStorageBackend): The backend class to be registered,
            which must be a subclass of :class:`BaseStorageBackend`.
        force (bool): Whether to override the backend if the name has already
            been registered. Defaults to False.
        prefixes (str or list[str] or tuple[str], optional): The prefix
            of the registered storage backend. Defaults to None.
    """
    global backends, prefix_to_backends

    if not isinstance(name, str):
        raise TypeError('the backend name should be a string, '
                        f'but got {type(name)}')

    if not inspect.isclass(backend):
        raise TypeError(f'backend should be a class, but got {type(backend)}')
    if not issubclass(backend, BaseStorageBackend):
        raise TypeError(
            f'backend {backend} is not a subclass of BaseStorageBackend')

    if name in backends and not force:
        raise ValueError(f'{name} is already registered as a storage backend, '
                         'add "force=True" if you want to override it')
    backends[name] = backend

    if prefixes is not None:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))

        for prefix in prefixes:
            if prefix in prefix_to_backends and not force:
                raise ValueError(
                    f'{prefix} is already registered as a storage backend,'
                    ' add "force=True" if you want to override it')

            prefix_to_backends[prefix] = backend


def register_backend(name: str,
                     backend: Optional[Type[BaseStorageBackend]] = None,
                     force: bool = False,
                     prefixes: Union[str, list, tuple, None] = None):
    """Register a backend.

    Args:
        name (str): The name of the registered backend.
        backend (class, optional): The backend class to be registered,
            which must be a subclass of :class:`BaseStorageBackend`.
            When this method is used as a decorator, backend is None.
            Defaults to None.
        force (bool): Whether to override the backend if the name has already
            been registered. Defaults to False.
        prefixes (str or list[str] or tuple[str], optional): The prefix
            of the registered storage backend. Defaults to None.

    This method can be used as a normal method or a decorator.

    Examples:

        >>> class NewBackend(BaseStorageBackend):
        ...     def get(self, filepath):
        ...         return filepath
        ...
        ...     def get_text(self, filepath):
        ...         return filepath
        >>> register_backend('new', NewBackend)

        >>> @register_backend('new')
        ... class NewBackend(BaseStorageBackend):
        ...     def get(self, filepath):
        ...         return filepath
        ...
        ...     def get_text(self, filepath):
        ...         return filepath
    """
    if backend is not None:
        _register_backend(name, backend, force=force, prefixes=prefixes)
        return

    def _register(backend_cls):
        _register_backend(name, backend_cls, force=force, prefixes=prefixes)
        return backend_cls

    return _register


register_backend('local', LocalBackend, prefixes='')
register_backend('memcached', MemcachedBackend)
register_backend('lmdb', LmdbBackend)
# To avoid breaking backward Compatibility, 's3' is also used as a
# prefix for PetrelBackend
register_backend('petrel', PetrelBackend, prefixes=['petrel', 's3'])
register_backend('http', HTTPBackend, prefixes=['http', 'https'])

def _parse_uri_prefix(uri: Union[str, Path]) -> str:
    # assert is_filepath(uri)
    uri = str(uri)
    # if uri does not contains '://', the uri will be handled by
    # LocalBackend by default
    if '://' not in uri:
        return ''
    else:
        prefix, _ = uri.split('://')
        # In the case of PetrelBackend, the prefix may contain the cluster
        # name like clusterName:s3://path/of/your/file
        if ':' in prefix:
            _, prefix = prefix.split(':')
        return prefix
    
def _get_file_backend(prefix: str, backend_args: dict):
    # backend name has a higher priority
    if 'backend' in backend_args:
        # backend_args should not be modified
        backend_args_bak = backend_args.copy()
        backend_name = backend_args_bak.pop('backend')
        backend = backends[backend_name](**backend_args_bak)
    else:
        backend = prefix_to_backends[prefix](**backend_args)
    return backend

def get_file_backend(
    uri: Union[str, Path, None] = None,
    *,
    backend_args: Optional[dict] = None,
    enable_singleton: bool = False,
):
    global backend_instances

    if backend_args is None:
        backend_args = {}

    if uri is None and 'backend' not in backend_args:
        raise ValueError(
            'uri should not be None when "backend" does not exist in '
            'backend_args')

    if uri is not None:
        prefix = _parse_uri_prefix(uri)
    else:
        prefix = ''

    if enable_singleton:
        # TODO: whether to pass sort_key to json.dumps
        unique_key = f'{prefix}:{json.dumps(backend_args)}'
        if unique_key in backend_instances:
            return backend_instances[unique_key]

        backend = _get_file_backend(prefix, backend_args)
        backend_instances[unique_key] = backend
        return backend
    else:
        backend = _get_file_backend(prefix, backend_args)
        return backend

def join_path(
    filepath: Union[str, Path],
    *filepaths: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.join_path(filepath, *filepaths)

def is_abs(path: str) -> bool:
    """Check if path is an absolute path in different backends.

    Args:
        path (str): path of directory or file.

    Returns:
        bool: whether path is an absolute path.
    """
    if osp.isabs(path) or path.startswith(('http://', 'https://', 's3://')):
        return True
    else:
        return False

TemplateArrayType = Union[np.ndarray, torch.Tensor, list, tuple, int, float]

class ArrayConverter:
    """Utility class for data-type agnostic processing.

    Args:
        template_array (np.ndarray or torch.Tensor or list or tuple or int or
            float, optional): Template array. Defaults to None.
    """
    SUPPORTED_NON_ARRAY_TYPES = (int, float, np.int8, np.int16, np.int32,
                                 np.int64, np.uint8, np.uint16, np.uint32,
                                 np.uint64, np.float16, np.float32, np.float64)

    def __init__(self,
                 template_array: Optional[TemplateArrayType] = None) -> None:
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array: TemplateArrayType) -> None:
        """Set template array.

        Args:
            array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Template array.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.
        """
        self.array_type = type(array)
        self.is_num = False
        self.device = 'cpu'

        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                print('The following list cannot be converted to a numpy '
                      f'array of supported dtype:\n{array}')
                raise
        elif isinstance(array, (int, float)):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(
                f'Template type {self.array_type} is not supported.')

    def convert(
        self,
        input_array: TemplateArrayType,
        target_type: Optional[Type] = None,
        target_array: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert input array to target data type.

        Args:
            input_array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Input array.
            target_type (Type, optional): Type to which input array is
                converted. It should be `np.ndarray` or `torch.Tensor`.
                Defaults to None.
            target_array (np.ndarray or torch.Tensor, optional): Template array
                to which input array is converted. Defaults to None.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.

        Returns:
            np.ndarray or torch.Tensor: The converted array.
        """
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                print('The input cannot be converted to a single-type numpy '
                      f'array:\n{input_array}')
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, \
            'must specify a target'
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), \
                'invalid target type'
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                # default dtype is float32
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                # default dtype is float32, device is 'cpu'
                converted_array = torch.tensor(
                    input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), \
                'invalid target array type'
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(
                    target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(
        self, input_array: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor, int, float]:
        """Recover input type to original array type.

        Args:
            input_array (np.ndarray or torch.Tensor): Input array.

        Returns:
            np.ndarray or torch.Tensor or int or float: Converted array.
        """
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), \
            'invalid input array type'
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(
                input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array


from inspect import getfullargspec
def array_converter(to_torch: bool = True,
                    apply_to: Tuple[str, ...] = tuple(),
                    template_arg_name_: Optional[str] = None,
                    recover: bool = True) -> Callable:
    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)

            func_name = func.__name__

            arg_spec = getfullargspec(func)

            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)

            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}

            all_arg_names = arg_names + kwonly_arg_names

            # in case there are args in the form of *args
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []

            # template argument data type is used for all array-like arguments
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_

            if template_arg_name not in all_arg_names:
                raise ValueError(f'{template_arg_name} is not among the '
                                 f'argument list of function {func_name}')

            # inspect apply_to
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(
                        f'{arg_to_apply} is not an argument of {func_name}')

            new_args = []
            new_kwargs = {}

            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray

            # non-keyword arguments
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(
                        converter.convert(
                            input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)

                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value

            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values

            # keyword arguments and non-keyword arguments using default value
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=kwargs[arg_name],
                            target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(
                            input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]

            # add nameless args provided by *args (if exists)
            new_args += nameless_args

            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data,
                                                         tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data

            if recover:
                return recursive_recover(return_values)
            else:
                return return_values

        return new_func

    return array_converter_wrapper

@array_converter(apply_to=('points', 'angles'))
def rotation_3d_in_axis(
    points: Union[np.ndarray, Tensor],
    angles: Union[np.ndarray, Tensor, float],
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor], np.ndarray,
           Tensor]:
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and \
        points.shape[0] == angles.shape[0], 'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(
                f'axis should in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new
    
class BasePoints:

    def __init__(self,
                 tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
                 points_dim: int = 3,
                 attribute_dims: Optional[dict] = None) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, points_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, \
            ('The points dimension must be 2 and the length of the last '
             f'dimension must be {points_dim}, but got points with shape '
             f'{tensor.shape}.')

        self.tensor = tensor.clone()
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self) -> Tensor:
        """Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the coordinates of each point.

        Args:
            tensor (Tensor or np.ndarray): Coordinates of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with height of each point in shape
        (N, )."""
        if self.attribute_dims is not None and \
                'height' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['height']]
        else:
            return None

    @height.setter
    def height(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the height of each point.

        Args:
            tensor (Tensor or np.ndarray): Height of each point with shape
                (N, ).
        """
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and \
                'height' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['height']] = tensor
        else:
            # add height attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with color of each point in shape
        (N, 3)."""
        if self.attribute_dims is not None and \
                'color' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['color']]
        else:
            return None

    @color.setter
    def color(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the color of each point.

        Args:
            tensor (Tensor or np.ndarray): Color of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn('point got color value beyond [0, 255]')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and \
                'color' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['color']] = tensor
        else:
            # add color attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(
                dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self) -> torch.Size:
        """torch.Size: Shape of points."""
        return self.tensor.shape

    def shuffle(self) -> Tensor:
        """Shuffle the points.

        Returns:
            Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self,
               rotation: Union[Tensor, np.ndarray, float],
               axis: Optional[int] = None) -> Tensor:
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (Tensor or np.ndarray or float): Rotation matrix or angle.
            axis (int, optional): Axis to rotate at. Defaults to None.

        Returns:
            Tensor: Rotation matrix.
        """
        if not isinstance(rotation, Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, \
            f'invalid rotation shape {rotation.shape}'

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rotated_points, rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, :3][None], rotation, axis=axis, return_mat=True)
            self.tensor[:, :3] = rotated_points.squeeze(0)
            rot_mat_T = rot_mat_T.squeeze(0)
        else:
            # rotation.numel() == 9
            self.tensor[:, :3] = self.tensor[:, :3] @ rotation
            rot_mat_T = rotation

        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction: str = 'horizontal') -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """
        pass

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        """Translate points with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size 3
                or nx3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and \
                trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(
                f'Unsupported translation vector of shape {trans_vector.shape}'
            )
        self.tensor[:, :3] += trans_vector

    def in_range_3d(
            self, point_range: Union[Tensor, np.ndarray,
                                     Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > point_range[0])
                          & (self.tensor[:, 1] > point_range[1])
                          & (self.tensor[:, 2] > point_range[2])
                          & (self.tensor[:, 0] < point_range[3])
                          & (self.tensor[:, 1] < point_range[4])
                          & (self.tensor[:, 2] < point_range[5]))
        return in_range_flags

    @property
    def bev(self) -> Tensor:
        """Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 1]]

    def in_range_bev(
            self, point_range: Union[Tensor, np.ndarray,
                                     Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point in order of (x_min, y_min, x_max, y_max).

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = ((self.bev[:, 0] > point_range[0])
                          & (self.bev[:, 1] > point_range[1])
                          & (self.bev[:, 0] < point_range[2])
                          & (self.bev[:, 1] < point_range[3]))
        return in_range_flags

    @abstractmethod
    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor,
                                          np.ndarray]] = None) -> 'BasePoints':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Point mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type in the
            ``dst`` mode.
        """
        pass

    def scale(self, scale_factor: float) -> None:
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(
            self, item: Union[int, tuple, slice, np.ndarray,
                              Tensor]) -> 'BasePoints':
        """
        Args:
            item (int or tuple or slice or np.ndarray or Tensor): Index of
                points.

        Note:
            The following usage are allowed:

            1. `new_points = points[3]`: Return a `Points` that contains only
               one point.
            2. `new_points = points[2:10]`: Return a slice of points.
            3. `new_points = points[vector]`: Whether vector is a
               torch.BoolTensor with `length = len(points)`. Nonzero elements
               in the vector will be selected.
            4. `new_points = points[3:11, vector]`: Return a slice of points
               and attribute dims.
            5. `new_points = points[4:12, 2]`: Return a slice of points with
               single attribute.

            Note that the returned Points might share storage with this Points,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of :class:`BasePoints` after
            indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                points_dim=self.points_dim,
                attribute_dims=self.attribute_dims)
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] \
                    if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]

            keep_dims = list(
                set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(
                        set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f'Invalid slice {item}!')

        assert p.dim() == 2, \
            f'Indexing on Points with {item} failed to return a matrix!'
        return original_type(
            p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self) -> int:
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, points_list: Sequence['BasePoints']) -> 'BasePoints':
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (Sequence[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        cat_points = cls(
            torch.cat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].points_dim,
            attribute_dims=points_list[0].attribute_dims)
        return cat_points

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args,
           **kwargs) -> 'BasePoints':
        """Convert current points to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new points object on the specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def cpu(self) -> 'BasePoints':
        """Convert current points to cpu device.

        Returns:
            :obj:`BasePoints`: A new points object on the cpu device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cpu(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def cuda(self, *args, **kwargs) -> 'BasePoints':
        """Convert current points to cuda device.

        Returns:
            :obj:`BasePoints`: A new points object on the cuda device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def clone(self) -> 'BasePoints':
        """Clone the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def detach(self) -> 'BasePoints':
        """Detach the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.detach(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    @property
    def device(self) -> torch.device:
        """torch.device: The device of the points are on."""
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        """Yield a point as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A point of shape (points_dim, ).
        """
        yield from self.tensor

    def new_point(
        self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]
    ) -> 'BasePoints':
        """Create a new point object with data.

        The new point and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``, the object's
            other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(
            new_tensor,
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

class LiDARPoints(BasePoints):
    """Points of instances in LIDAR coordinates.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The points
            data with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...).
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self,
                 tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
                 points_dim: int = 3,
                 attribute_dims: Optional[dict] = None) -> None:
        super(LiDARPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction: str = 'horizontal') -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1] = -self.tensor[:, 1]
        elif bev_direction == 'vertical':
            self.tensor[:, 0] = -self.tensor[:, 0]

    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor,
                                          np.ndarray]] = None) -> 'BasePoints':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Point mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type in the
            ``dst`` mode.
        """
        from mmdet3d.structures.bbox_3d import Coord3DMode
        return Coord3DMode.convert_point(
            point=self, src=Coord3DMode.LIDAR, dst=dst, rt_mat=rt_mat)
    
class BaseInstance3DBoxes:
    YAW_AXIS: int = 0

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 0.5, 0)
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, \
            ('The box dimension must be 2 and the length of the last '
             f'dimension must be {box_dim}, but got boxes with shape '
             f'{tensor.shape}.')

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding 0 as
            # a fake yaw and set with_yaw to False
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def shape(self) -> torch.Size:
        """torch.Size: Shape of boxes."""
        return self.tensor.shape

    @property
    def volume(self) -> Tensor:
        """Tensor: A vector with volume of each box in shape (N, )."""
        return self.tensor[:, 3] * self.tensor[:, 4] * self.tensor[:, 5]

    @property
    def dims(self) -> Tensor:
        """Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self) -> Tensor:
        """Tensor: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def height(self) -> Tensor:
        """Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 5]

    @property
    def top_height(self) -> Tensor:
        """Tensor: A vector with top height of each box in shape (N, )."""
        return self.bottom_height + self.height

    @property
    def bottom_height(self) -> Tensor:
        """Tensor: A vector with bottom height of each box in shape (N, )."""
        return self.tensor[:, 2]

    @property
    def center(self) -> Tensor:
        """Calculate the center of all the boxes.

        Note:
            In MMDetection3D's convention, the bottom center is usually taken
            as the default center.

            The relative position of the centers in different kinds of boxes
            are different, e.g., the relative center of a boxes is
            (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar. It is
            recommended to use ``bottom_center`` or ``gravity_center`` for
            clearer usage.

        Returns:
            Tensor: A tensor with center of each box in shape (N, 3).
        """
        return self.bottom_center

    @property
    def bottom_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def gravity_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self) -> Tensor:
        """Tensor: A tensor with 8 corners of each box in shape (N, 8, 3)."""
        pass

    @property
    def bev(self) -> Tensor:
        """Tensor: 2D BEV box of each box with rotation in XYWHR format, in
        shape (N, 5)."""
        return self.tensor[:, [0, 1, 3, 4, 6]]

    @property
    def nearest_bev(self) -> Tensor:
        """Tensor: A tensor of 2D BEV box of each box without rotation."""
        # Obtain BEV boxes with rotation in XYWHR format
        bev_rotated_boxes = self.bev
        # convert the rotation to a valid range
        rotations = bev_rotated_boxes[:, -1]
        normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

        # find the center of boxes
        conditions = (normed_rotations > np.pi / 4)[..., None]
        bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:,
                                                                [0, 1, 3, 2]],
                                  bev_rotated_boxes[:, :4])

        centers = bboxes_xywh[:, :2]
        dims = bboxes_xywh[:, 2:]
        bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
        return bev_boxes

    def in_range_bev(
            self, box_range: Union[Tensor, np.ndarray,
                                   Sequence[float]]) -> Tensor:
        """Check whether the boxes are in the given range.

        Args:
            box_range (Tensor or np.ndarray or Sequence[float]): The range of
                box in order of (x_min, y_min, x_max, y_max).

        Note:
            The original implementation of SECOND checks whether boxes in a
            range by checking whether the points are in a convex polygon, we
            reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each box is inside the
            reference range.
        """
        in_range_flags = ((self.bev[:, 0] > box_range[0])
                          & (self.bev[:, 1] > box_range[1])
                          & (self.bev[:, 0] < box_range[2])
                          & (self.bev[:, 1] < box_range[3]))
        return in_range_flags

    @abstractmethod
    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        pass

    @abstractmethod
    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        pass

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        """Translate boxes with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size
                1x3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor[:, :3] += trans_vector

    def in_range_3d(
            self, box_range: Union[Tensor, np.ndarray,
                                   Sequence[float]]) -> Tensor:
        """Check whether the boxes are in the given range.

        Args:
            box_range (Tensor or np.ndarray or Sequence[float]): The range of
                box (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        gravity_center = self.gravity_center
        in_range_flags = ((gravity_center[:, 0] > box_range[0])
                          & (gravity_center[:, 1] > box_range[1])
                          & (gravity_center[:, 2] > box_range[2])
                          & (gravity_center[:, 0] < box_range[3])
                          & (gravity_center[:, 1] < box_range[4])
                          & (gravity_center[:, 2] < box_range[5]))
        return in_range_flags

    @abstractmethod
    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        pass

    def scale(self, scale_factor: float) -> None:
        """Scale the box with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the boxes.
        """
        self.tensor[:, :6] *= scale_factor
        self.tensor[:, 7:] *= scale_factor  # velocity

    def limit_yaw(self, offset: float = 0.5, period: float = np.pi) -> None:
        """Limit the yaw to a given period and offset.

        Args:
            offset (float): The offset of the yaw. Defaults to 0.5.
            period (float): The expected period. Defaults to np.pi.
        """
        self.tensor[:, 6] = limit_period(self.tensor[:, 6], offset, period)

    def nonempty(self, threshold: float = 0.0) -> Tensor:
        """Find boxes that are non-empty.

        A box is considered empty if either of its side is no larger than
        threshold.

        Args:
            threshold (float): The threshold of minimal sizes. Defaults to 0.0.

        Returns:
            Tensor: A binary vector which represents whether each box is empty
            (False) or non-empty (True).
        """
        box = self.tensor
        size_x = box[..., 3]
        size_y = box[..., 4]
        size_z = box[..., 5]
        keep = ((size_x > threshold)
                & (size_y > threshold) & (size_z > threshold))
        return keep

    def __getitem__(
            self, item: Union[int, slice, np.ndarray,
                              Tensor]) -> 'BaseInstance3DBoxes':
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.

        Note:
            The following usage are allowed:

            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.

            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                box_dim=self.box_dim,
                with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def __len__(self) -> int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, boxes_list: Sequence['BaseInstance3DBoxes']
            ) -> 'BaseInstance3DBoxes':
        """Concatenate a list of Boxes into a single Boxes.

        Args:
            boxes_list (Sequence[:obj:`BaseInstance3DBoxes`]): List of boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: The concatenated boxes.
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(box, cls) for box in boxes_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned boxes never share storage with input
        cat_boxes = cls(
            torch.cat([b.tensor for b in boxes_list], dim=0),
            box_dim=boxes_list[0].box_dim,
            with_yaw=boxes_list[0].with_yaw)
        return cat_boxes

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args,
           **kwargs) -> 'BaseInstance3DBoxes':
        """Convert current boxes to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the specific
            device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs),
            box_dim=self.box_dim,
            with_yaw=self.with_yaw)

    def cpu(self) -> 'BaseInstance3DBoxes':
        """Convert current boxes to cpu device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cpu device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cpu(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def cuda(self, *args, **kwargs) -> 'BaseInstance3DBoxes':
        """Convert current boxes to cuda device.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the cuda device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs),
            box_dim=self.box_dim,
            with_yaw=self.with_yaw)

    def clone(self) -> 'BaseInstance3DBoxes':
        """Clone the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def detach(self) -> 'BaseInstance3DBoxes':
        """Detach the boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties as
            self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.detach(), box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def device(self) -> torch.device:
        """torch.device: The device of the boxes are on."""
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        """Yield a box as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A box of shape (box_dim, ).
        """
        yield from self.tensor

    @classmethod
    def height_overlaps(cls, boxes1: 'BaseInstance3DBoxes',
                        boxes2: 'BaseInstance3DBoxes') -> Tensor:
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.

        Returns:
            Tensor: Calculated height overlap of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), \
            '"boxes1" and "boxes2" should be in the same type, ' \
            f'but got {type(boxes1)} and {type(boxes2)}.'

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        heighest_of_bottom = torch.max(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.min(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(lowest_of_top - heighest_of_bottom, min=0)
        return overlaps_h

    @classmethod
    def overlaps(cls,
                 boxes1: 'BaseInstance3DBoxes',
                 boxes2: 'BaseInstance3DBoxes',
                 mode: str = 'iou') -> Tensor:
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`BaseInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`BaseInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str): Mode of iou calculation. Defaults to 'iou'.

        Returns:
            Tensor: Calculated 3D overlap of the boxes.
        """
        assert isinstance(boxes1, BaseInstance3DBoxes)
        assert isinstance(boxes2, BaseInstance3DBoxes)
        assert type(boxes1) == type(boxes2), \
            '"boxes1" and "boxes2" should be in the same type, ' \
            f'but got {type(boxes1)} and {type(boxes2)}.'

        assert mode in ['iou', 'iof']

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        # height overlap
        overlaps_h = cls.height_overlaps(boxes1, boxes2)

        # Restrict the min values of W and H to avoid memory overflow in
        # ``box_iou_rotated``.
        boxes1_bev, boxes2_bev = boxes1.bev, boxes2.bev
        boxes1_bev[:, 2:4] = boxes1_bev[:, 2:4].clamp(min=1e-4)
        boxes2_bev[:, 2:4] = boxes2_bev[:, 2:4].clamp(min=1e-4)

        # bev overlap
        iou2d = box_iou_rotated(boxes1_bev, boxes2_bev)
        areas1 = (boxes1_bev[:, 2] * boxes1_bev[:, 3]).unsqueeze(1).expand(
            rows, cols)
        areas2 = (boxes2_bev[:, 2] * boxes2_bev[:, 3]).unsqueeze(0).expand(
            rows, cols)
        overlaps_bev = iou2d * (areas1 + areas2) / (1 + iou2d)

        # 3d overlaps
        overlaps_3d = overlaps_bev.to(boxes1.device) * overlaps_h

        volume1 = boxes1.volume.view(-1, 1)
        volume2 = boxes2.volume.view(1, -1)

        if mode == 'iou':
            # the clamp func is used to avoid division of 0
            iou3d = overlaps_3d / torch.clamp(
                volume1 + volume2 - overlaps_3d, min=1e-8)
        else:
            iou3d = overlaps_3d / torch.clamp(volume1, min=1e-8)

        return iou3d

    def new_box(
        self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]
    ) -> 'BaseInstance3DBoxes':
        """Create a new box object with data.

        The new box and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new bbox object with ``data``, the
            object's other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(
            new_tensor, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def points_in_boxes_part(
            self,
            points: Tensor,
            boxes_override: Optional[Tensor] = None) -> Tensor:
        """Find the box in which each point is.

        Args:
            points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
                are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (Tensor, optional): Boxes to override `self.tensor`.
                Defaults to None.

        Note:
            If a point is enclosed by multiple boxes, the index of the first
            box will be returned.

        Returns:
            Tensor: The index of the first box that each point is in with shape
            (M, ). Default value is -1 (if the point is not enclosed by any
            box).
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor

        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1

        boxes = boxes.to(points_clone.device).unsqueeze(0)
        box_idx = points_in_boxes_part(points_clone, boxes)

        return box_idx.squeeze(0)

    def points_in_boxes_all(self,
                            points: Tensor,
                            boxes_override: Optional[Tensor] = None) -> Tensor:
        """Find all boxes in which each point is.

        Args:
            points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
                are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (Tensor, optional): Boxes to override `self.tensor`.
                Defaults to None.

        Returns:
            Tensor: A tensor indicating whether a point is in a box with shape
            (M, T). T is the number of boxes. Denote this tensor as A, it the
            m^th point is in the t^th box, then `A[m, t] == 1`, otherwise
            `A[m, t] == 0`.
        """
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor

        points_clone = points.clone()[..., :3]
        if points_clone.dim() == 2:
            points_clone = points_clone.unsqueeze(0)
        else:
            assert points_clone.dim() == 3 and points_clone.shape[0] == 1

        boxes = boxes.to(points_clone.device).unsqueeze(0)
        box_idxs_of_pts = points_in_boxes_all(points_clone, boxes)

        return box_idxs_of_pts.squeeze(0)

    def points_in_boxes(self,
                        points: Tensor,
                        boxes_override: Optional[Tensor] = None) -> Tensor:
        warnings.warn('DeprecationWarning: points_in_boxes is a deprecated '
                      'method, please consider using points_in_boxes_part.')
        return self.points_in_boxes_part(points, boxes_override)

    def points_in_boxes_batch(
            self,
            points: Tensor,
            boxes_override: Optional[Tensor] = None) -> Tensor:
        warnings.warn('DeprecationWarning: points_in_boxes_batch is a '
                      'deprecated method, please consider using '
                      'points_in_boxes_all.')
        return self.points_in_boxes_all(points, boxes_override)


@unique
class Box3DMode(IntEnum):
    """Enum of different ways to represent a box.

    Coordinates in LiDAR:

    .. code-block:: none

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in Camera:

    .. code-block:: none

                z front
               /
              /
             0 ------> x right
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth:

    .. code-block:: none

        up z
           ^   y front
           |  /
           | /
           0 ------> x right

    The relative coordinate of bottom center in a DEPTH box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAM = 1
    DEPTH = 2

    @staticmethod
    def convert(
        box: Union[Sequence[float], np.ndarray, Tensor, BaseInstance3DBoxes],
        src: 'Box3DMode',
        dst: 'Box3DMode',
        rt_mat: Optional[Union[np.ndarray, Tensor]] = None,
        with_yaw: bool = True,
        correct_yaw: bool = False
    ) -> Union[Sequence[float], np.ndarray, Tensor, BaseInstance3DBoxes]:
        """Convert boxes from ``src`` mode to ``dst`` mode.

        Args:
            box (Sequence[float] or np.ndarray or Tensor or
                :obj:`BaseInstance3DBoxes`): Can be a k-tuple, k-list or an Nxk
                array/tensor.
            src (:obj:`Box3DMode`): The source box mode.
            dst (:obj:`Box3DMode`): The target box mode.
            rt_mat (np.ndarray or Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            with_yaw (bool): If ``box`` is an instance of
                :obj:`BaseInstance3DBoxes`, whether or not it has a yaw angle.
                Defaults to True.
            correct_yaw (bool): If the yaw is rotated by rt_mat.
                Defaults to False.

        Returns:
            Sequence[float] or np.ndarray or Tensor or
            :obj:`BaseInstance3DBoxes`: The converted box of the same type.
        """
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        is_Instance3DBoxes = isinstance(box, BaseInstance3DBoxes)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 7, (
                'Box3DMode.convert takes either a k-tuple/list or '
                'an Nxk array/tensor, where k >= 7')
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            elif is_Instance3DBoxes:
                arr = box.tensor.clone()
            else:
                arr = box.clone()

        if is_Instance3DBoxes:
            with_yaw = box.with_yaw

        # convert box from `src` mode to `dst` mode.
        x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
        if with_yaw:
            yaw = arr[..., 6:7]
        if src == Box3DMode.LIDAR and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([
                        torch.cos(yaw),
                        torch.sin(yaw),
                        torch.zeros_like(yaw)
                    ],
                                           dim=1)
                else:
                    yaw = -yaw - np.pi / 2
                    yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.CAM and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([
                        torch.cos(-yaw),
                        torch.zeros_like(yaw),
                        torch.sin(-yaw)
                    ],
                                           dim=1)
                else:
                    yaw = -yaw - np.pi / 2
                    yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([
                        torch.cos(yaw),
                        torch.sin(yaw),
                        torch.zeros_like(yaw)
                    ],
                                           dim=1)
                else:
                    yaw = -yaw
        elif src == Box3DMode.CAM and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([
                        torch.cos(-yaw),
                        torch.zeros_like(yaw),
                        torch.sin(-yaw)
                    ],
                                           dim=1)
                else:
                    yaw = -yaw
        elif src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([
                        torch.cos(yaw),
                        torch.sin(yaw),
                        torch.zeros_like(yaw)
                    ],
                                           dim=1)
                else:
                    yaw = yaw + np.pi / 2
                    yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([
                        torch.cos(yaw),
                        torch.sin(yaw),
                        torch.zeros_like(yaw)
                    ],
                                           dim=1)
                else:
                    yaw = yaw - np.pi / 2
                    yaw = limit_period(yaw, period=np.pi * 2)
        else:
            raise NotImplementedError(
                f'Conversion from Box3DMode {src} to {dst} '
                'is not supported yet')

        if not isinstance(rt_mat, Tensor):
            rt_mat = arr.new_tensor(rt_mat)
        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat(
                [arr[..., :3], arr.new_ones(arr.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = arr[..., :3] @ rt_mat.t()

        # Note: we only use rotation in rt_mat
        # so don't need to extend yaw_vector
        if with_yaw and correct_yaw:
            rot_yaw_vector = yaw_vector @ rt_mat[:3, :3].t()
            if dst == Box3DMode.CAM:
                yaw = torch.atan2(-rot_yaw_vector[:, [2]], rot_yaw_vector[:,
                                                                          [0]])
            elif dst in [Box3DMode.LIDAR, Box3DMode.DEPTH]:
                yaw = torch.atan2(rot_yaw_vector[:, [1]], rot_yaw_vector[:,
                                                                         [0]])
            yaw = limit_period(yaw, period=np.pi * 2)

        if with_yaw:
            remains = arr[..., 7:]
            arr = torch.cat([xyz[..., :3], xyz_size, yaw, remains], dim=-1)
        else:
            remains = arr[..., 6:]
            arr = torch.cat([xyz[..., :3], xyz_size, remains], dim=-1)

        # convert arr to the original type
        original_type = type(box)
        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        elif is_Instance3DBoxes:
            if dst == Box3DMode.CAM:
                target_type = CameraInstance3DBoxes
            elif dst == Box3DMode.LIDAR:
                target_type = LiDARInstance3DBoxes
            elif dst == Box3DMode.DEPTH:
                target_type = DepthInstance3DBoxes
            else:
                raise NotImplementedError(
                    f'Conversion to {dst} through {original_type} '
                    'is not supported yet')
            return target_type(arr, box_dim=arr.size(-1), with_yaw=with_yaw)
        else:
            return arr

class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                 up z    x front (yaw=0)
                                    ^   ^
                                    |  /
                                    | /
        (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2. The yaw is 0 at
    the positive direction of x axis, and increases from the positive direction
    of x to the positive direction of y.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS = 2

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                                           up z
                            front x           ^
                                 /            |
                                /             |
                  (x1, y0, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
            left y <------- + ----------- + (x0, y1, z0)
                (x0, y0, z0)

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0.5, 0)
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(
            corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        if not isinstance(angle, Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=self.YAW_AXIS,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        elif bev_direction == 'vertical':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi

        if points is not None:
            assert isinstance(points, (Tensor, np.ndarray, BasePoints))
            if isinstance(points, (Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 1] = -points[:, 1]
                elif bev_direction == 'vertical':
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        # from .box_3d_mode import Box3DMode
        return Box3DMode.convert(
            box=self,
            src=Box3DMode.LIDAR,
            dst=dst,
            rt_mat=rt_mat,
            correct_yaw=correct_yaw)

    def enlarged_box(
            self, extra_width: Union[float, Tensor]) -> 'LiDARInstance3DBoxes':
        """Enlarge the length, width and height of boxes.

        Args:
            extra_width (float or Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

class DepthInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in DEPTH coordinates.

    Coordinates in Depth:

    .. code-block:: none

        up z    y front (yaw=0.5*pi)
           ^   ^
           |  /
           | /
           0 ------> x right (yaw=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2. The yaw is 0 at
    the positive direction of x axis, and increases from the positive direction
    of x to the positive direction of y.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS = 2

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                                        up z
                         front y           ^
                              /            |
                             /             |
               (x0, y1, z1) + -----------  + (x1, y1, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                         |  /      .   |  /
                         | / origin    | /
            (x0, y0, z0) + ----------- + --------> right x
                                       (x1, y0, z0)

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 0.5, 0)
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(
            corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        if not isinstance(angle, Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=self.YAW_AXIS,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        if self.with_yaw:
            self.tensor[:, 6] += angle
        else:
            # for axis-aligned boxes, we take the new
            # enclosing axis-aligned boxes after rotation
            corners_rot = self.corners @ rot_mat_T
            new_x_size = corners_rot[..., 0].max(
                dim=1, keepdim=True)[0] - corners_rot[..., 0].min(
                    dim=1, keepdim=True)[0]
            new_y_size = corners_rot[..., 1].max(
                dim=1, keepdim=True)[0] - corners_rot[..., 1].min(
                    dim=1, keepdim=True)[0]
            self.tensor[:, 3:5] = torch.cat((new_x_size, new_y_size), dim=-1)

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        In Depth coordinates, it flips the x (horizontal) or y (vertical) axis.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == 'vertical':
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (Tensor, np.ndarray, BasePoints))
            if isinstance(points, (Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 0] = -points[:, 0]
                elif bev_direction == 'vertical':
                    points[:, 1] = -points[:, 1]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode
        return Box3DMode.convert(
            box=self,
            src=Box3DMode.DEPTH,
            dst=dst,
            rt_mat=rt_mat,
            correct_yaw=correct_yaw)

    def enlarged_box(
            self, extra_width: Union[float, Tensor]) -> 'DepthInstance3DBoxes':
        """Enlarge the length, width and height of boxes.

        Args:
            extra_width (float or Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`DepthInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    def get_surface_line_center(self) -> Tuple[Tensor, Tensor]:
        """Compute surface and line center of bounding boxes.

        Returns:
            Tuple[Tensor, Tensor]: Surface and line center of bounding boxes.
        """
        obj_size = self.dims
        center = self.gravity_center.view(-1, 1, 3)
        batch_size = center.shape[0]

        rot_sin = torch.sin(-self.yaw)
        rot_cos = torch.cos(-self.yaw)
        rot_mat_T = self.yaw.new_zeros(tuple(list(self.yaw.shape) + [3, 3]))
        rot_mat_T[..., 0, 0] = rot_cos
        rot_mat_T[..., 0, 1] = -rot_sin
        rot_mat_T[..., 1, 0] = rot_sin
        rot_mat_T[..., 1, 1] = rot_cos
        rot_mat_T[..., 2, 2] = 1

        # Get the object surface center
        offset = obj_size.new_tensor([[0, 0, 1], [0, 0, -1], [0, 1, 0],
                                      [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
        offset = offset.view(1, 6, 3) / 2
        surface_3d = (offset *
                      obj_size.view(batch_size, 1, 3).repeat(1, 6, 1)).reshape(
                          -1, 3)

        # Get the object line center
        offset = obj_size.new_tensor([[1, 0, 1], [-1, 0, 1], [0, 1, 1],
                                      [0, -1, 1], [1, 0, -1], [-1, 0, -1],
                                      [0, 1, -1], [0, -1, -1], [1, 1, 0],
                                      [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
        offset = offset.view(1, 12, 3) / 2

        line_3d = (offset *
                   obj_size.view(batch_size, 1, 3).repeat(1, 12, 1)).reshape(
                       -1, 3)

        surface_rot = rot_mat_T.repeat(6, 1, 1)
        surface_3d = torch.matmul(surface_3d.unsqueeze(-2),
                                  surface_rot).squeeze(-2)
        surface_center = center.repeat(1, 6, 1).reshape(-1, 3) + surface_3d

        line_rot = rot_mat_T.repeat(12, 1, 1)
        line_3d = torch.matmul(line_3d.unsqueeze(-2), line_rot).squeeze(-2)
        line_center = center.repeat(1, 12, 1).reshape(-1, 3) + line_3d

        return surface_center, line_center


class CameraInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in CAM coordinates.

    Coordinates in Camera:

    .. code-block:: none

                z front (yaw=-0.5*pi)
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1. The yaw is 0 at
    the positive direction of x axis, and decreases from the positive direction
    of x to the positive direction of z.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
            data with shape (N, box_dim).
        box_dim (int): Number of the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw). Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation. If False, the
            value of yaw will be set to 0 as minmax boxes. Defaults to True.
        origin (Tuple[float]): Relative position of the box origin.
            Defaults to (0.5, 1.0, 0.5). This will guide the box be converted
            to (0.5, 1.0, 0.5) mode.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """
    YAW_AXIS = 1

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 1.0, 0.5)
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, \
            ('The box dimension must be 2 and the length of the last '
             f'dimension must be {box_dim}, but got boxes with shape '
             f'{tensor.shape}.')

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding 0 as
            # a fake yaw and set with_yaw to False
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 1.0, 0.5):
            dst = self.tensor.new_tensor((0.5, 1.0, 0.5))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    @property
    def height(self) -> Tensor:
        """Tensor: A vector with height of each box in shape (N, )."""
        return self.tensor[:, 4]

    @property
    def top_height(self) -> Tensor:
        """Tensor: A vector with top height of each box in shape (N, )."""
        # the positive direction is down rather than up
        return self.bottom_height - self.height

    @property
    def bottom_height(self) -> Tensor:
        """Tensor: A vector with bottom height of each box in shape (N, )."""
        return self.tensor[:, 1]

    @property
    def local_yaw(self) -> Tensor:
        """Tensor: A vector with local yaw of each box in shape (N, ).
        local_yaw equals to alpha in kitti, which is commonly used in monocular
        3D object detection task, so only :obj:`CameraInstance3DBoxes` has the
        property."""
        yaw = self.yaw
        loc = self.gravity_center
        local_yaw = yaw2local(yaw, loc)

        return local_yaw

    @property
    def gravity_center(self) -> Tensor:
        """Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, [0, 2]] = bottom_center[:, [0, 2]]
        gravity_center[:, 1] = bottom_center[:, 1] - self.tensor[:, 4] * 0.5
        return gravity_center

    @property
    def corners(self) -> Tensor:
        """Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                         front z
                              /
                             /
               (x0, y0, z1) + -----------  + (x1, y0, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                         |  /      .   |  /
                         | / origin    | /
            (x0, y1, z0) + ----------- + -------> right x
                         |             (x1, y1, z0)
                         |
                         v
                    down y

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 1, 0.5)
        corners_norm = corners_norm - dims.new_tensor([0.5, 1, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        corners = rotation_3d_in_axis(
            corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners

    @property
    def bev(self) -> Tensor:
        """Tensor: 2D BEV box of each box with rotation in XYWHR format, in
        shape (N, 5)."""
        bev = self.tensor[:, [0, 2, 3, 5, 6]].clone()
        # positive direction of the gravity axis
        # in cam coord system points to the earth
        # so the bev yaw angle needs to be reversed
        bev[:, -1] = -bev[:, -1]
        return bev

    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray], Tuple[
            BasePoints, Tensor], None]:
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angle (Tensor or np.ndarray or float): Rotation angle or rotation
                matrix.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns None,
            otherwise it returns the rotated points and the rotation matrix
            ``rot_mat_T``.
        """
        if not isinstance(angle, Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=self.YAW_AXIS,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[2, 0]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """Flip the boxes in BEV along given BEV direction.

        In CAM coordinates, it flips the x (horizontal) or z (vertical) axis.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (Tensor or np.ndarray or :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            Tensor or np.ndarray or :obj:`BasePoints` or None: When ``points``
            is None, the function returns None, otherwise it returns the
            flipped points.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        elif bev_direction == 'vertical':
            self.tensor[:, 2::7] = -self.tensor[:, 2::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (Tensor, np.ndarray, BasePoints))
            if isinstance(points, (Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 0] = -points[:, 0]
                elif bev_direction == 'vertical':
                    points[:, 2] = -points[:, 2]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    @classmethod
    def height_overlaps(cls, boxes1: 'CameraInstance3DBoxes',
                        boxes2: 'CameraInstance3DBoxes') -> Tensor:
        """Calculate height overlaps of two boxes.

        Note:
            This function calculates the height overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`CameraInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`CameraInstance3DBoxes`): Boxes 2 contain M boxes.

        Returns:
            Tensor: Calculated height overlap of the boxes.
        """
        assert isinstance(boxes1, CameraInstance3DBoxes)
        assert isinstance(boxes2, CameraInstance3DBoxes)

        boxes1_top_height = boxes1.top_height.view(-1, 1)
        boxes1_bottom_height = boxes1.bottom_height.view(-1, 1)
        boxes2_top_height = boxes2.top_height.view(1, -1)
        boxes2_bottom_height = boxes2.bottom_height.view(1, -1)

        # positive direction of the gravity axis
        # in cam coord system points to the earth
        heighest_of_bottom = torch.min(boxes1_bottom_height,
                                       boxes2_bottom_height)
        lowest_of_top = torch.max(boxes1_top_height, boxes2_top_height)
        overlaps_h = torch.clamp(heighest_of_bottom - lowest_of_top, min=0)
        return overlaps_h

    def convert_to(self,
                   dst: int,
                   rt_mat: Optional[Union[Tensor, np.ndarray]] = None,
                   correct_yaw: bool = False) -> 'BaseInstance3DBoxes':
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Box mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            correct_yaw (bool): Whether to convert the yaw angle to the target
                coordinate. Defaults to False.

        Returns:
            :obj:`BaseInstance3DBoxes`: The converted box of the same type in
            the ``dst`` mode.
        """
        from .box_3d_mode import Box3DMode

        # TODO: always set correct_yaw=True
        return Box3DMode.convert(
            box=self,
            src=Box3DMode.CAM,
            dst=dst,
            rt_mat=rt_mat,
            correct_yaw=correct_yaw)

    def points_in_boxes_part(
            self,
            points: Tensor,
            boxes_override: Optional[Tensor] = None) -> Tensor:
        """Find the box in which each point is.

        Args:
            points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
                are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (Tensor, optional): Boxes to override `self.tensor`.
                Defaults to None.

        Returns:
            Tensor: The index of the first box that each point is in with shape
            (M, ). Default value is -1 (if the point is not enclosed by any
            box).
        """
        from .coord_3d_mode import Coord3DMode

        points_lidar = Coord3DMode.convert(points, Coord3DMode.CAM,
                                           Coord3DMode.LIDAR)
        if boxes_override is not None:
            boxes_lidar = boxes_override
        else:
            boxes_lidar = Coord3DMode.convert(
                self.tensor,
                Coord3DMode.CAM,
                Coord3DMode.LIDAR,
                is_point=False)

        box_idx = super().points_in_boxes_part(points_lidar, boxes_lidar)
        return box_idx

    def points_in_boxes_all(self,
                            points: Tensor,
                            boxes_override: Optional[Tensor] = None) -> Tensor:
        """Find all boxes in which each point is.

        Args:
            points (Tensor): Points in shape (1, M, 3) or (M, 3), 3 dimensions
                are (x, y, z) in LiDAR or depth coordinate.
            boxes_override (Tensor, optional): Boxes to override `self.tensor`.
                Defaults to None.

        Returns:
            Tensor: The index of all boxes in which each point is with shape
            (M, T).
        """
        from .coord_3d_mode import Coord3DMode

        points_lidar = Coord3DMode.convert(points, Coord3DMode.CAM,
                                           Coord3DMode.LIDAR)
        if boxes_override is not None:
            boxes_lidar = boxes_override
        else:
            boxes_lidar = Coord3DMode.convert(
                self.tensor,
                Coord3DMode.CAM,
                Coord3DMode.LIDAR,
                is_point=False)

        box_idx = super().points_in_boxes_all(points_lidar, boxes_lidar)
        return box_idx


def get_box_type(box_type: str) -> Tuple[type, int]:
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure. The valid value are "LiDAR",
            "Camera" and "Depth".

    Raises:
        ValueError: A ValueError is raised when ``box_type`` does not belong to
            the three valid types.

    Returns:
        tuple: Box type and box mode.
    """
    box_type_lower = box_type.lower()
    if box_type_lower == 'lidar':
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == 'camera':
        box_type_3d = CameraInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    elif box_type_lower == 'depth':
        box_type_3d = DepthInstance3DBoxes
        box_mode_3d = Box3DMode.DEPTH
    else:
        raise ValueError('Only "box_type" of "camera", "lidar", "depth" are '
                         f'supported, got {box_type}')

    return box_type_3d, box_mode_3d

class Compose:
    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
    
def force_full_init(old_func: Callable) -> Any:

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `full_init` method.
        if not hasattr(obj, 'full_init'):
            raise AttributeError(f'{type(obj)} does not have full_init '
                                 'method.')
        # If instance does not have `_fully_initialized` attribute or
        # `_fully_initialized` is False, call `full_init` and set
        # `_fully_initialized` to True
        if not getattr(obj, '_fully_initialized', False):
            # print_log(
            #     f'Attribute `_fully_initialized` is not defined in '
            #     f'{type(obj)} or `type(obj)._fully_initialized is '
            #     'False, `full_init` will be called and '
            #     f'{type(obj)}._fully_initialized will be set to True',
            #     logger='current',
            #     level=logging.WARNING)
            obj.full_init()  # type: ignore
            obj._fully_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper

class BaseDataset(Dataset):
    METAINFO: dict = dict()
    _fully_initialized: bool = False

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        self.ann_file = ann_file
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Join paths.
        self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            raw_data_info[prefix_key] = join_path(prefix,
                                                  raw_data_info[prefix_key])
        return raw_data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg. Defaults return all
        ``data_list``.

        If some ``data_list`` could be filtered according to specific logic,
        the subclass should override this method.

        Returns:
            list[int]: Filtered results.
        """
        return self.data_list

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``ClassBalancedDataset`` requires a subclass which implements this
        method.

        Args:
            idx (int): The index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        raise NotImplementedError(f'{type(self)} must implement `get_cat_ids` '
                                  'method')

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        from custom.load import load
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        # Meta information load from annotation file will not influence the
        # existed meta information load from `BaseDataset.METAINFO` and
        # `metainfo` arguments defined in constructor.
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    @classmethod
    def _load_metainfo(cls,
                       metainfo: Union[Mapping, Config, None] = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (Mapping or Config, optional): Meta information dict.
                If ``metainfo`` contains existed filename, it will be
                parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, (Mapping, Config)):
            raise TypeError('metainfo should be a Mapping or Config, '
                            f'but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information',
                        logger='current',
                        level=logging.WARNING)
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    @force_full_init
    def get_subset_(self, indices: Union[Sequence[int], int]) -> None:
        """The in-place version of ``get_subset`` to convert dataset to a
        subset of original dataset.

        This method will convert the original dataset to a subset of dataset.
        If type of indices is int, ``get_subset_`` will return a subdataset
        which contains the first or last few data information according to
        indices is positive or negative. If type of indices is a sequence of
        int, the subdataset will extract the data information according to
        the index given in indices.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> dataset.get_subset_(90)
              >>> len(dataset)
              90
              >>> # if type of indices is sequence, extract the corresponding
              >>> # index data information
              >>> dataset.get_subset_([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
              >>> len(dataset)
              10
              >>> dataset.get_subset_(-3)
              >>> len(dataset) # Get the latest few data information.
              3

        Args:
            indices (int or Sequence[int]): If type of indices is int, indices
                represents the first or last few data of dataset according to
                indices is positive or negative. If type of indices is
                Sequence, indices represents the target data information
                index of dataset.
        """
        # Get subset of data from serialized data or data information sequence
        # according to `self.serialize_data`.
        if self.serialize_data:
            self.data_bytes, self.data_address = \
                self._get_serialized_subset(indices)
        else:
            self.data_list = self._get_unserialized_subset(indices)

    @force_full_init
    def get_subset(self, indices: Union[Sequence[int], int]) -> 'BaseDataset':
        """Return a subset of dataset.

        This method will return a subset of original dataset. If type of
        indices is int, ``get_subset_`` will return a subdataset which
        contains the first or last few data information according to
        indices is positive or negative. If type of indices is a sequence of
        int, the subdataset will extract the information according to the index
        given in indices.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> subdataset = dataset.get_subset(90)
              >>> len(sub_dataset)
              90
              >>> # if type of indices is list, extract the corresponding
              >>> # index data information
              >>> subdataset = dataset.get_subset([0, 1, 2, 3, 4, 5, 6, 7,
              >>>                                  8, 9])
              >>> len(sub_dataset)
              10
              >>> subdataset = dataset.get_subset(-3)
              >>> len(subdataset) # Get the latest few data information.
              3

        Args:
            indices (int or Sequence[int]): If type of indices is int, indices
                represents the first or last few data of dataset according to
                indices is positive or negative. If type of indices is
                Sequence, indices represents the target data information
                index of dataset.

        Returns:
            BaseDataset: A subset of dataset.
        """
        # Get subset of data from serialized data or data information list
        # according to `self.serialize_data`. Since `_get_serialized_subset`
        # will recalculate the subset data information,
        # `_copy_without_annotation` will copy all attributes except data
        # information.
        sub_dataset = self._copy_without_annotation()
        # Get subset of dataset with serialize and unserialized data.
        if self.serialize_data:
            data_bytes, data_address = \
                self._get_serialized_subset(indices)
            sub_dataset.data_bytes = data_bytes.copy()
            sub_dataset.data_address = data_address.copy()
        else:
            data_list = self._get_unserialized_subset(indices)
            sub_dataset.data_list = copy.deepcopy(data_list)
        return sub_dataset

    def _get_serialized_subset(self, indices: Union[Sequence[int], int]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Get subset of serialized data information list.

        Args:
            indices (int or Sequence[int]): If type of indices is int,
                indices represents the first or last few data of serialized
                data information list. If type of indices is Sequence, indices
                represents the target data information index which consist of
                subset data information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of serialized data
            information.
        """
        sub_data_bytes: Union[List, np.ndarray]
        sub_data_address: Union[List, np.ndarray]
        if isinstance(indices, int):
            if indices >= 0:
                assert indices < len(self.data_address), \
                    f'{indices} is out of dataset length({len(self)}'
                # Return the first few data information.
                end_addr = self.data_address[indices - 1].item() \
                    if indices > 0 else 0
                # Slicing operation of `np.ndarray` does not trigger a memory
                # copy.
                sub_data_bytes = self.data_bytes[:end_addr]
                # Since the buffer size of first few data information is not
                # changed,
                sub_data_address = self.data_address[:indices]
            else:
                assert -indices <= len(self.data_address), \
                    f'{indices} is out of dataset length({len(self)}'
                # Return the last few data information.
                ignored_bytes_size = self.data_address[indices - 1]
                start_addr = self.data_address[indices - 1].item()
                sub_data_bytes = self.data_bytes[start_addr:]
                sub_data_address = self.data_address[indices:]
                sub_data_address = sub_data_address - ignored_bytes_size
        elif isinstance(indices, Sequence):
            sub_data_bytes = []
            sub_data_address = []
            for idx in indices:
                assert len(self) > idx >= -len(self)
                start_addr = 0 if idx == 0 else \
                    self.data_address[idx - 1].item()
                end_addr = self.data_address[idx].item()
                # Get data information by address.
                sub_data_bytes.append(self.data_bytes[start_addr:end_addr])
                # Get data information size.
                sub_data_address.append(end_addr - start_addr)
            # Handle indices is an empty list.
            if sub_data_bytes:
                sub_data_bytes = np.concatenate(sub_data_bytes)
                sub_data_address = np.cumsum(sub_data_address)
            else:
                sub_data_bytes = np.array([])
                sub_data_address = np.array([])
        else:
            raise TypeError('indices should be a int or sequence of int, '
                            f'but got {type(indices)}')
        return sub_data_bytes, sub_data_address  # type: ignore

    def _get_unserialized_subset(self, indices: Union[Sequence[int],
                                                      int]) -> list:
        """Get subset of data information list.

        Args:
            indices (int or Sequence[int]): If type of indices is int,
                indices represents the first or last few data of data
                information. If type of indices is Sequence, indices represents
                the target data information index which consist of subset data
                information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of data information.
        """
        if isinstance(indices, int):
            if indices >= 0:
                # Return the first few data information.
                sub_data_list = self.data_list[:indices]
            else:
                # Return the last few data information.
                sub_data_list = self.data_list[indices:]
        elif isinstance(indices, Sequence):
            # Return the data information according to given indices.
            sub_data_list = []
            for idx in indices:
                sub_data_list.append(self.data_list[idx])
        else:
            raise TypeError('indices should be a int or sequence of int, '
                            f'but got {type(indices)}')
        return sub_data_list

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        # Serialize data information list avoid making multiple copies of
        # `self.data_list` when iterate `import torch.utils.data.dataloader`
        # with multiple workers.
        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        # TODO Check if np.concatenate is necessary
        data_bytes = np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of
        # `self.data_info` when loading data multi-processes.
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    @force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def _copy_without_annotation(self, memo=dict()) -> 'BaseDataset':
        """Deepcopy for all attributes other than ``data_list``,
        ``data_address`` and ``data_bytes``.

        Args:
            memo: Memory dict which used to reconstruct complex object
                correctly.
        """
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            if key in ['data_list', 'data_address', 'data_bytes']:
                continue
            super(BaseDataset, other).__setattr__(key,
                                                  copy.deepcopy(value, memo))

        return other
    
@DATASETS.register_module()
class Det3DDataset(BaseDataset):
    """Base Class of 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.
    # TODO: doc link here for the standard data format

    Args:
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='velodyne', img='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input,
            it usually has following keys:

                - use_camera: bool
                - use_lidar: bool
            Defaults to dict(use_lidar=True, use_camera=False).
        default_cam_key (str, optional): The default camera name adopted.
            Defaults to None.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates, usually for
              outdoor point cloud 3d detection.
            - 'Depth': Box in depth coordinates, usually for
              indoor point cloud 3d detection.
            - 'Camera': Box in camera coordinates, usually
              for vision-based 3d detection.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        load_eval_anns (bool): Whether to load annotations in test_mode,
            the annotation will be save in `eval_ann_infos`, which can be
            used in Evaluator. Defaults to True.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        show_ins_var (bool): For debug purpose. Whether to show variation
            of the number of instances before and after through pipeline.
            Defaults to False.
    """

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = None,
                 box_type_3d: dict = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 show_ins_var: bool = False,
                 **kwargs) -> None:
        self.backend_args = backend_args
        self.filter_empty_gt = filter_empty_gt
        self.load_eval_anns = load_eval_anns
        _default_modality_keys = ('use_lidar', 'use_camera')
        if modality is None:
            modality = dict()

        # Defaults to False if not specify
        for key in _default_modality_keys:
            if key not in modality:
                modality[key] = False
        self.modality = modality
        self.default_cam_key = default_cam_key
        assert self.modality['use_lidar'] or self.modality['use_camera'], (
            'Please specify the `modality` (`use_lidar` '
            f', `use_camera`) for {self.__class__.__name__}')

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        if metainfo is not None and 'classes' in metainfo:
            # we allow to train on subset of self.METAINFO['classes']
            # map unselected labels to -1
            self.label_mapping = {
                i: -1
                for i in range(len(self.METAINFO['classes']))
            }
            self.label_mapping[-1] = -1
            for label_idx, name in enumerate(metainfo['classes']):
                ori_label = self.METAINFO['classes'].index(name)
                self.label_mapping[ori_label] = label_idx

            self.num_ins_per_cat = [0] * len(metainfo['classes'])
        else:
            self.label_mapping = {
                i: i
                for i in range(len(self.METAINFO['classes']))
            }
            self.label_mapping[-1] = -1

            self.num_ins_per_cat = [0] * len(self.METAINFO['classes'])

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

        # can be accessed by other component in runner
        self.metainfo['box_type_3d'] = box_type_3d
        self.metainfo['label_mapping'] = self.label_mapping

        if not kwargs.get('lazy_init', False):
            # used for showing variation of the number of instances before and
            # after through the pipeline
            self.show_ins_var = show_ins_var

            # show statistics of this dataset
            print('-' * 30, 'current')
            print(
                f'The length of {"test" if self.test_mode else "training"} dataset: {len(self)}',  # noqa: E501
                'current')
            content_show = [['category', 'number']]
            for label, num in enumerate(self.num_ins_per_cat):
                cat_name = self.metainfo['classes'][label]
                content_show.append([cat_name, num])
            # table = AsciiTable(content_show)
            # print(
            #     f'The number of instances per category in the dataset:\n{table.table}',  # noqa: E501
            #     'current')

    def _remove_dontcare(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        -1 indicates dontcare in MMDet3d.

        Args:
            ann_info (dict): Dict of annotation infos. The
                instance with label `-1` will be removed.

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        filter_mask = ann_info['gt_labels_3d'] > -1
        for key in ann_info.keys():
            if key != 'instances':
                img_filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                img_filtered_annotations[key] = ann_info[key]
        return img_filtered_annotations

    def get_ann_info(self, index: int) -> dict:
        """Get annotation info according to the given index.

        Use index to get the corresponding annotations, thus the
        evalhook could use this api.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information.
        """
        data_info = self.get_data_info(index)
        # test model
        if 'ann_info' not in data_info:
            ann_info = self.parse_ann_info(data_info)
        else:
            ann_info = data_info['ann_info']

        return ann_info

    def parse_ann_info(self, info: dict) -> Union[dict, None]:
        """Process the `instances` in data info to `ann_info`.

        In `Custom3DDataset`, we simply concatenate all the field
        in `instances` to `np.ndarray`, you can do the specific
        process in subclass. You have to convert `gt_bboxes_3d`
        to different coordinates according to the task.

        Args:
            info (dict): Info dict.

        Returns:
            dict or None: Processed `ann_info`.
        """
        # add s or gt prefix for most keys after concat
        # we only process 3d annotations here, the corresponding
        # 2d annotation process is in the `LoadAnnotations3D`
        # in `transforms`
        name_mapping = {
            'bbox_label_3d': 'gt_labels_3d',
            'bbox_label': 'gt_bboxes_labels',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels',
            'velocity': 'velocities',
        }
        instances = info['instances']
        # empty gt
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = dict()
            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label
                if 'label' in ann_name and ann_name != 'attr_label':
                    temp_anns = [
                        self.label_mapping[item] for item in temp_anns
                    ]
                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name

                if 'label' in ann_name:
                    temp_anns = np.array(temp_anns).astype(np.int64)
                elif ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns
            ann_info['instances'] = info['instances']

            for label in ann_info['gt_labels_3d']:
                if label != -1:
                    self.num_ins_per_cat[label] += 1

        return ann_info

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """

        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])

            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']
            if 'lidar_sweeps' in info:
                for sweep in info['lidar_sweeps']:
                    file_suffix = sweep['lidar_points']['lidar_path'].split(
                        os.sep)[-1]
                    if 'samples' in sweep['lidar_points']['lidar_path']:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['pts'], file_suffix)
                    else:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['sweeps'], file_suffix)

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get('img', '')
                    img_info['img_path'] = osp.join(cam_prefix,
                                                    img_info['img_path'])
            if self.default_cam_key is not None:
                info['img_path'] = info['images'][
                    self.default_cam_key]['img_path']
                if 'lidar2cam' in info['images'][self.default_cam_key]:
                    info['lidar2cam'] = np.array(
                        info['images'][self.default_cam_key]['lidar2cam'])
                if 'cam2img' in info['images'][self.default_cam_key]:
                    info['cam2img'] = np.array(
                        info['images'][self.default_cam_key]['cam2img'])
                if 'lidar2img' in info['images'][self.default_cam_key]:
                    info['lidar2img'] = np.array(
                        info['images'][self.default_cam_key]['lidar2img'])
                else:
                    info['lidar2img'] = info['cam2img'] @ info['lidar2cam']

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info

    def _show_ins_var(self, old_labels: np.ndarray,
                      new_labels: torch.Tensor) -> None:
        """Show variation of the number of instances before and after through
        the pipeline.

        Args:
            old_labels (np.ndarray): The labels before through the pipeline.
            new_labels (torch.Tensor): The labels after through the pipeline.
        """
        ori_num_per_cat = dict()
        for label in old_labels:
            if label != -1:
                cat_name = self.metainfo['classes'][label]
                ori_num_per_cat[cat_name] = ori_num_per_cat.get(cat_name,
                                                                0) + 1
        new_num_per_cat = dict()
        for label in new_labels:
            if label != -1:
                cat_name = self.metainfo['classes'][label]
                new_num_per_cat[cat_name] = new_num_per_cat.get(cat_name,
                                                                0) + 1
        content_show = [['category', 'new number', 'ori number']]
        for cat_name, num in ori_num_per_cat.items():
            new_num = new_num_per_cat.get(cat_name, 0)
            content_show.append([cat_name, new_num, num])
        table = AsciiTable(content_show)
        print_log(
            'The number of instances per category after and before '
            f'through pipeline:\n{table.table}', 'current')

    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None

        example = self.pipeline(input_dict)

        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(
                    example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None

        if self.show_ins_var:
            if 'ann_info' in ori_input_dict:
                self._show_ins_var(
                    ori_input_dict['ann_info']['gt_labels_3d'],
                    example['data_samples'].gt_instances_3d.labels_3d)
            else:
                print_log(
                    "'ann_info' is not in the input dict. It's probably that "
                    'the data is not in training mode',
                    'current',
                    level=30)

        return example

    def get_cat_ids(self, idx: int) -> Set[int]:
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``CBGSDataset`` or ``ClassBalancedDataset``requires a subclass
        which implements this method.

        Args:
            idx (int): The index of data.

        Returns:
            set[int]: All categories in the sample of specified index.
        """
        info = self.get_data_info(idx)
        gt_labels = info['ann_info']['gt_labels_3d'].tolist()
        return set(gt_labels)
    
@DATASETS.register_module()
class NuScenesDataset(Det3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
                to convert to the FOV-based data type to support image-based
                detector.
            - 'fov_image_based': Only load the instances inside the default
                cam, and need to convert to the FOV-based data type to support
                image-based detector.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        with_velocity (bool): Whether to include velocity prediction
            into the experiments. Defaults to True.
        use_valid_flag (bool): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    METAINFO = {
        'classes':
        ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'version':
        'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
            (220, 20, 60),  # Crimson
            (255, 61, 99),  # Red
            (0, 0, 230),  # Blue
            (47, 79, 79),  # Darkslategrey
            (112, 128, 144),  # Slategrey
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 modality: dict = dict(
                     use_camera=False,
                     use_lidar=True,
                 ),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 with_velocity: bool = True,
                 use_valid_flag: bool = False,
                 **kwargs) -> None:
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity

        # TODO: Redesign multi-view data process in the future
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type

        assert box_type_3d.lower() in ('lidar', 'camera')
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            modality=modality,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

    def _filter_with_mask(self, ann_info: dict) -> dict:
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos.

        Returns:
            dict: Annotations after filtering.
        """
        filtered_annotations = {}
        if self.use_valid_flag:
            filter_mask = ann_info['bbox_3d_isvalid']
        else:
            filter_mask = ann_info['num_lidar_pts'] > 0
        for key in ann_info.keys():
            if key != 'instances':
                filtered_annotations[key] = (ann_info[key][filter_mask])
            else:
                filtered_annotations[key] = ann_info[key]
        return filtered_annotations

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is not None:

            ann_info = self._filter_with_mask(ann_info)

            if self.with_velocity:
                gt_bboxes_3d = ann_info['gt_bboxes_3d']
                gt_velocities = ann_info['velocities']
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocities],
                                              axis=-1)
                ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        else:
            # empty instance
            ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['attr_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # TODO: Unify the coordinates
        if self.load_type in ['fov_image_based', 'mv_image_based']:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5))
        else:
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d

        return ann_info

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.load_type == 'mv_image_based':
            data_list = []
            if self.modality['use_lidar']:
                info['lidar_points']['lidar_path'] = \
                    osp.join(
                        self.data_prefix.get('pts', ''),
                        info['lidar_points']['lidar_path'])

            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        if cam_id in self.data_prefix:
                            cam_prefix = self.data_prefix[cam_id]
                        else:
                            cam_prefix = self.data_prefix.get('img', '')
                        img_info['img_path'] = osp.join(
                            cam_prefix, img_info['img_path'])

            for idx, (cam_id, img_info) in enumerate(info['images'].items()):
                camera_info = dict()
                camera_info['images'] = dict()
                camera_info['images'][cam_id] = img_info
                if 'cam_instances' in info and cam_id in info['cam_instances']:
                    camera_info['instances'] = info['cam_instances'][cam_id]
                else:
                    camera_info['instances'] = []
                # TODO: check whether to change sample_idx for 6 cameras
                #  in one frame
                camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
                camera_info['token'] = info['token']
                camera_info['ego2global'] = info['ego2global']

                if not self.test_mode:
                    # used in traing
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info['eval_ann_info'] = \
                        self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            data_info = super().parse_data_info(info)
            return data_info