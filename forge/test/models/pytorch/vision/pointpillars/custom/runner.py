# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import os.path as osp
import pickle
import platform
import time
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union, Iterator, Tuple, Type, TypeVar
from collections import OrderedDict, namedtuple
import inspect
import torch
import torch.nn as nn
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from custom.imports import Config, ConfigDict, master_only, is_seq_of, get_dist_info
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
T = TypeVar('T')
import random
from custom.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, METRICS)
from custom.base import BaseDataElement, BaseModel
from custom.default_scope import DefaultScope
from custom.preprocessor import Det3DDataPreprocessor
TORCH_VERSION = torch.__version__

DATA_BATCH = Optional[Union[dict, tuple, list]]
VisBackendsType = Union[List[Union[List, BaseDataElement]], BaseDataElement,
                        dict, None]
import threading
_lock = threading.RLock()
from torch.distributed import ProcessGroup
from torch import distributed as torch_dist
from packaging.version import parse

class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: Dict[str, Callable] = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger='current'):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Defaults to None
            logger (str): The logger for message. Defaults to 'current'.

        Returns:
            dict o(r OrderedDict: The loaded checkpoint.
        """
        # breakpoint()
        checkpoint_loader = cls._get_checkpoint_loader(filename)
        # class_name = checkpoint_loader.__name__
        class_name = "nuscenes_pointpillar"
        print(
            f'Loads checkpoint by {class_name[10:]} backend from path: '
            f'{filename}')
        return checkpoint_loader(filename, map_location)
    
@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(filename, map_location):
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('http://', 'https://'))
def load_from_http(filename,
                   map_location=None,
                   model_dir=None,
                   progress=os.isatty(0)):
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (string, optional): directory in which to save the object,
            Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(
            filename,
            model_dir=model_dir,
            map_location=map_location,
            progress=progress)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = load_url(
                filename,
                model_dir=model_dir,
                map_location=map_location,
                progress=progress)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes='pavi://')
def load_from_pavi(filename, map_location=None):
    """load checkpoint through the file path prefixed with pavi. In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Args:
        filename (str): checkpoint file path with pavi prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    assert filename.startswith('pavi://'), \
        f'Expected filename startswith `pavi://`, but get {filename}'
    model_path = filename[7:]

    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')

    model = modelcloud.get(model_path)
    with TemporaryDirectory() as tmp_dir:
        downloaded_file = osp.join(tmp_dir, model.name)
        model.download(downloaded_file)
        checkpoint = torch.load(downloaded_file, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(
    prefixes=[r'(\S+\:)?s3://', r'(\S+\:)?petrel://'])
def load_from_ceph(filename, map_location=None, backend='petrel'):
    """load checkpoint through the file path prefixed with s3.  In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Args:
        filename (str): checkpoint file path with s3 prefix
        map_location (str, optional): Same as :func:`torch.load`.
        backend (str, optional): The storage backend type.
            Defaults to 'petrel'.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    file_backend = get_file_backend(
        filename, backend_args={'backend': backend})
    with io.BytesIO(file_backend.get(filename)) as buffer:
        checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('modelzoo://', 'torchvision://'))
def load_from_torchvision(filename, map_location=None):
    """load checkpoint through the file path prefixed with modelzoo or
    torchvision.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    model_urls = get_torchvision_models()
    if filename.startswith('modelzoo://'):
        print_log(
            'The URL scheme of "modelzoo://" is deprecated, please '
            'use "torchvision://" instead',
            logger='current',
            level=logging.WARNING)
        model_name = filename[11:]
    else:
        model_name = filename[14:]
    return load_from_http(model_urls[model_name], map_location=map_location)


@CheckpointLoader.register_scheme(prefixes=('open-mmlab://', 'openmmlab://'))
def load_from_openmmlab(filename, map_location=None):
    """load checkpoint through the file path prefixed with open-mmlab or
    openmmlab.

    Args:
        filename (str): checkpoint file path with open-mmlab or
        openmmlab prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_external_models()
    prefix_str = 'open-mmlab://'
    if filename.startswith(prefix_str):
        model_name = filename[13:]
    else:
        model_name = filename[12:]
        prefix_str = 'openmmlab://'

    deprecated_urls = get_deprecated_model_names()
    if model_name in deprecated_urls:
        print_log(
            f'{prefix_str}{model_name} is deprecated in favor '
            f'of {prefix_str}{deprecated_urls[model_name]}',
            logger='current',
            level=logging.WARNING)
        model_name = deprecated_urls[model_name]
    model_url = model_urls[model_name]
    # check if is url
    if model_url.startswith(('http://', 'https://')):
        checkpoint = load_from_http(model_url, map_location=map_location)
    else:
        filename = osp.join(_get_mmengine_home(), model_url)
        if not osp.isfile(filename):
            raise FileNotFoundError(f'{filename} can not be found.')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes='mmcls://')
def load_from_mmcls(filename, map_location=None):
    """load checkpoint through the file path prefixed with mmcls.

    Args:
        filename (str): checkpoint file path with mmcls prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_mmcls_models()
    model_name = filename[8:]
    checkpoint = load_from_http(
        model_urls[model_name], map_location=map_location)
    checkpoint = _process_mmcls_checkpoint(checkpoint)
    return checkpoint


def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str, optional): Same as :func:`torch.load`.
           Defaults to None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
        OrderedDict storing model weights or a dict containing other
        information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


def _load_checkpoint_with_prefix(prefix, filename, map_location=None):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict

EnhancedBatchInputs = List[Union[torch.Tensor, List[torch.Tensor]]]
# multi-batch data samples processed by different augmentations from the same
# batch. The inner list stands for different augmentations and the outer list
# stands for batch.
EnhancedBatchDataSamples = List[List[BaseDataElement]]
DATA_BATCH = Union[Dict[str, Union[EnhancedBatchInputs,
                                   EnhancedBatchDataSamples]], tuple, dict]
MergedDataSamples = List[BaseDataElement]

class _IncompatibleKeys(
        namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__

@MODELS.register_module()
class BaseTTAModel(BaseModel):
    def __init__(
        self,
        module: Union[dict, nn.Module],
        data_preprocessor: Union[dict, nn.Module, None] = None,
    ):
        super().__init__()
        if isinstance(module, nn.Module):
            self.module = module
        elif isinstance(module, dict):
            if data_preprocessor is not None:
                module['data_preprocessor'] = data_preprocessor
            self.module = MODELS.build(module)
        else:
            raise TypeError('The type of module should be a `nn.Module` '
                            f'instance or a dict, but got {module}')
        assert hasattr(self.module, 'test_step'), (
            'Model wrapped by BaseTTAModel must implement `test_step`!')

    @abstractmethod
    def merge_preds(self, data_samples_list: EnhancedBatchDataSamples) \
            -> MergedDataSamples:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (EnhancedBatchDataSamples): List of predictions
                of all enhanced data.

        Returns:
            List[BaseDataElement]: Merged prediction.
        """

    def test_step(self, data):
        """Get predictions of each enhanced data, a multiple predictions.

        Args:
            data (DataBatch): Enhanced data batch sampled from dataloader.

        Returns:
            MergedDataSamples: Merged prediction.
        """
        data_list: Union[List[dict], List[list]]
        if isinstance(data, dict):
            num_augs = len(data[next(iter(data))])
            data_list = [{key: value[idx]
                          for key, value in data.items()}
                         for idx in range(num_augs)]
        elif isinstance(data, (tuple, list)):
            num_augs = len(data[0])
            data_list = [[_data[idx] for _data in data]
                         for idx in range(num_augs)]
        else:
            raise TypeError('data given by dataLoader should be a dict, '
                            f'tuple or a list, but got {type(data)}')

        predictions = []
        for data in data_list:  # type: ignore
            predictions.append(self.module.test_step(data))
        return self.merge_preds(list(zip(*predictions)))  # type: ignore

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """``BaseTTAModel.forward`` should not be called."""
        raise NotImplementedError(
            '`BaseTTAModel.forward` will not be called during training or'
            'testing. Please call `test_step` instead. If you want to use'
            '`BaseTTAModel.forward`, please implement this method')
    
def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    missing_keys = []
    err_msg = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, local_state_dict, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_model_wrapper(module) or isinstance(module, BaseTTAModel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(local_state_dict, prefix, local_metadata,
                                     True, missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {
                    k: v
                    for k, v in local_state_dict.items()
                    if k.startswith(child_prefix)
                }
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if hasattr(module, '_load_state_dict_post_hooks'):
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    'Hooks registered with '
                    '``register_load_state_dict_post_hook`` are not expected '
                    'to return new values, if incompatible_keys need to be '
                    'modified, it should be done inplace.')

    load(module, state_dict)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print_log(err_msg, logger=logger, level=logging.WARNING)

def _load_checkpoint_to_model(model,
                              checkpoint,
                              strict=False,
                              logger=None,
                              revise_keys=[(r'^module\.', '')]):

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    return _load_checkpoint_to_model(model, checkpoint, strict, logger,
                                     revise_keys)


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    # stash metadata to put in state_dict later
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    state_dict = apply_to(state_dict, lambda x: hasattr(x, 'cpu'),
                          lambda x: x.cpu())
    state_dict._metadata = metadata
    return state_dict


# @deprecated_function(
#     since='0.3.0',
#     removed_in='0.5.0',
#     instructions='`_save_to_state_dict` will be deprecated in the future, '
#     'please use `nn.Module._save_to_state_dict` directly.')
def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
        keep_vars (bool): Whether to keep the variable property of the
            parameters.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        if buf is not None and name not in module._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Defaults to False.

    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_model_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    module._save_to_state_dict(destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(checkpoint,
                    filename,
                    file_client_args=None,
                    backend_args=None):
    """Save checkpoint to file.

    Args:
        checkpoint (dict): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            `backend_args` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.
    """
    if file_client_args is not None:
        print_log(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            logger='current',
            level=logging.WARNING)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set '
                'at the same time.')

    if filename.startswith('pavi://'):
        if file_client_args is not None or backend_args is not None:
            raise ValueError(
                '"file_client_args" or "backend_args" should be "None" if '
                'filename starts with "pavi://"')
        try:
            from pavi import exception, modelcloud
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except exception.NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        file_client = FileClient.infer_client(file_client_args, filename)
        if file_client_args is None:
            file_backend = get_file_backend(
                filename, backend_args=backend_args)
        else:
            file_backend = file_client

        with io.BytesIO() as f:
            torch.save(checkpoint, f)
            file_backend.put(f.getvalue(), filename)


def find_latest_checkpoint(path: str) -> Optional[str]:
    """Find the latest checkpoint from the given path.

    Refer to https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/checkpoint.py  # noqa: E501

    Args:
        path(str): The path to find checkpoints.

    Returns:
        str or None: File path of the latest checkpoint.
    """
    save_file = osp.join(path, 'last_checkpoint')
    last_saved: Optional[str]
    if os.path.exists(save_file):
        with open(save_file) as f:
            last_saved = f.read().strip()
    else:
        print_log('Did not find last_checkpoint to be resumed.')
        last_saved = None
    return last_saved

    
def _load_checkpoint(filename, map_location=None, logger=None):
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)

def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)

def get_rank(group: Optional[ProcessGroup] = None) -> int:
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0
    
def get_comm_device(group: Optional[ProcessGroup] = None) -> torch.device:
    return torch.device('cpu')
    
def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()

def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()

def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1
    
def sync_random_seed(group: Optional[ProcessGroup] = None) -> int:
    seed = np.random.randint(2**31)
    if get_world_size(group) == 1:
        return seed

    if group is None:
        group = get_default_group()

    backend_device = get_comm_device(group)

    if get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(backend_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(backend_device)

    torch_dist.broadcast(random_num, src=0, group=group)

    return random_num.item()

def set_random_seed(seed: Optional[int] = None,
                    deterministic: bool = False,
                    diff_rank_seed: bool = False) -> int:
    if seed is None:
        seed = sync_random_seed()

    if diff_rank_seed:
        rank = get_rank()
        seed += rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # if is_cuda_available():
    #     torch.cuda.manual_seed_all(seed)
    # elif is_musa_available():
    #     torch.musa.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print_log(
                'torch.backends.cudnn.benchmark is going to be set as '
                '`False` to cause cuDNN to deterministically select an '
                'algorithm',
                logger='current',
                level=logging.WARNING)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
            torch.use_deterministic_algorithms(True)
    return seed

def get_device() -> str:
    """Returns the currently existing device type.

    Returns:
        str: cuda | npu | mlu | mps | musa | cpu.
    """
    DEVICE = 'cpu'
    return DEVICE

def _release_lock() -> None:
    """Release the module-level lock acquired by calling _accquire_lock()."""
    if _lock:
        _lock.release()

def _accquire_lock() -> None:
    """Acquire the module-level lock for serializing access to shared data.

    This should be released with _release_lock().
    """
    if _lock:
        _lock.acquire()

def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """Helper function to convert all `SyncBatchNorm` (SyncBN) and
    `mmcv.ops.sync_bn.SyncBatchNorm`(MMSyncBN) layers in the model to
    `BatchNormXd` layers.

    Adapted from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.

    Returns:
        module_output: The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]

    # if mmcv_full_available():
    #     from mmcv.ops import SyncBatchNorm
    #     module_checklist.append(SyncBatchNorm)

    if isinstance(module, tuple(module_checklist)):
        module_output = _BatchNormXd(module.num_features, module.eps,
                                     module.momentum, module.affine,
                                     module.track_running_stats)
        if module.affine:
            # no_grad() may not be needed here but
            # just to be consistent with `convert_sync_batchnorm()`
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        # qconfig exists in quantized models
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        # Some custom modules or 3rd party implemented modules may raise an
        # error when calling `add_module`. Therefore, try to catch the error
        # and do not raise it. See https://github.com/open-mmlab/mmengine/issues/638 # noqa: E501
        # for more details.
        try:
            module_output.add_module(name, revert_sync_batchnorm(child))
        except Exception:
            print(
                F'Failed to convert {child} from SyncBN to BN!')
    del module
    return module_output

class BaseOptimWrapper(metaclass=ABCMeta):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        if len(optimizer.param_groups) > 1:
            self.base_param_settings = {
                'params': torch.tensor([0.0], dtype=torch.float)
            }
            self.base_param_settings.update(**self.optimizer.defaults)
        else:
            self.base_param_settings = None  # type: ignore

    @abstractmethod
    def update_params(self, *args, **kwargs):
        """Update parameters in :attr:`optimizer`."""

    @abstractmethod
    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation."""

    @abstractmethod
    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``."""

    @abstractmethod
    def step(self, **kwargs):
        """Call the step method of optimizer."""

    def state_dict(self) -> dict:
        """A wrapper of ``Optimizer.state_dict``."""
        state_dict = self.optimizer.state_dict()
        if self.base_param_settings is not None:
            state_dict['base_param_settings'] = self.base_param_settings
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """A wrapper of ``Optimizer.load_state_dict``. load the state dict of
        :attr:`optimizer`.

        Provide unified ``load_state_dict`` interface compatible with automatic
        mixed precision training. Subclass can overload this method to
        implement the required logic. For example, the state dictionary of
        GradScaler should be loaded when training with ``torch.cuda.amp``.

        Args:
            state_dict (dict): The state dictionary of :attr:`optimizer`.
        """
        base_param_settings = state_dict.pop('base_param_settings', None)

        if base_param_settings is not None:
            self.base_param_settings = base_param_settings

        # load state_dict of optimizer
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self) -> List[dict]:
        """A wrapper of ``Optimizer.param_groups``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        if self.base_param_settings is not None:
            return self.optimizer.param_groups + [self.base_param_settings]
        else:
            return self.optimizer.param_groups

    @property
    def defaults(self) -> dict:
        """A wrapper of ``Optimizer.defaults``.

        Make OptimizeWrapper compatible with :class:`_ParamScheduler`.

        Returns:
             dict: the ``param_groups`` of :attr:`optimizer`.
        """
        return self.optimizer.defaults

    def get_lr(self):
        """Get the learning rate of the optimizer.

        Provide unified interface to get learning rate of optimizer.

        Returns:
            Dict[str, List[float]]:
            param_groups learning rate of the optimizer.
        """
        res = {}
        if self.base_param_settings is not None:
            res['base_lr'] = [self.base_param_settings['lr']]

        res['lr'] = [group['lr'] for group in self.optimizer.param_groups]

        return res

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of the optimizer.

        Provide unified interface to get momentum of optimizer.

        Returns:
            Dict[str, List[float]]: Momentum of the optimizer.
        """
        momentum = []
        for group in self.optimizer.param_groups:
            # Get momentum of SGD.
            if 'momentum' in group.keys():
                momentum.append(group['momentum'])
            # Get momentum of Adam.
            elif 'betas' in group.keys():
                momentum.append(group['betas'][0])
            else:
                momentum.append(0)
        return dict(momentum=momentum)
    
INF = int(1e9)

OptimizerType = Union[BaseOptimWrapper, Optimizer]
class _ParamScheduler:
    def __init__(self,
                 optimizer: OptimizerType,
                 param_name: str,
                 begin: int = 0,
                 end: int = INF,
                 last_step: int = -1,
                 by_epoch: bool = True,
                 verbose: bool = False):

        # Attach optimizer
        if not isinstance(optimizer, (Optimizer, BaseOptimWrapper)):
            raise TypeError('``optimizer`` should be an Optimizer,'
                            'but got {}'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.param_name = param_name

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        self.by_epoch = by_epoch

        assert isinstance(last_step, int) and last_step >= -1
        # Initialize valid step count and base values
        if last_step == -1:
            for group in optimizer.param_groups:
                # If the param is never be scheduled, record the current value
                # as the initial value.
                group.setdefault(f'initial_{param_name}', group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f'initial_{param_name}' not in group:
                    raise KeyError(
                        f"param 'initial_{param_name}' is not specified "
                        'in param_groups[{}] when resuming an optimizer'.
                        format(i))
        self.base_values = [
            group[f'initial_{param_name}'] for group in optimizer.param_groups
        ]
        self.last_step = last_step

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method: Callable):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)  # type: ignore
            # Get the unbound method for the same purpose.
            func = method.__func__  # type: ignore
            cls = instance_ref().__class__  # type: ignore
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._global_step += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore
            return wrapper

        # add counter to optimizer
        self.optimizer.step = with_counter(self.optimizer.step)  # type: ignore
        self.optimizer._global_step = -1  # type: ignore

        self._global_step = -1
        self.verbose = verbose

        self.step()

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not
        the optimizer.

        Returns:
            dict: scheduler state.
        """
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """Return the last computed value by current scheduler.

        Returns:
            list: A list of the last computed value of the optimizer's
            ``param_group``.
        """
        return self._last_value

    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        raise NotImplementedError

    def print_value(self, is_verbose: bool, group: int, value: float):
        """Display the current parameter value.

        Args:
            is_verbose (bool): Whether to print the value.
            group (int): The index of the current ``param_group``.
            value (float): The parameter value.
        """
        if is_verbose:
            print_log(
                f'Adjusting parameter value of group {group} to {value:.4e}.',
                logger='current')

    def step(self):
        """Adjusts the parameter value of each parameter group based on the
        specified schedule."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._global_step == 0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn(
                    'Seems like `optimizer.step()` has been overridden after '
                    'parameter value scheduler initialization. Please, make '
                    'sure to call `optimizer.step()` before '
                    '`scheduler.step()`. See more details at '
                    'https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)

            # Just check if there were two first scheduler.step() calls
            # before optimizer.step()
            elif self.optimizer._global_step < 0:
                warnings.warn(
                    'Detected call of `scheduler.step()` before '
                    '`optimizer.step()`. In PyTorch 1.1.0 and later, you '
                    'should call them in the opposite order: '
                    '`optimizer.step()` before `scheduler.step()`. '
                    'Failure to do this will result in PyTorch skipping '
                    'the first value of the parameter value schedule. '
                    'See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate',  # noqa: E501
                    UserWarning)
        self._global_step += 1

        # Compute parameter value per param group in the effective range
        if self.begin <= self._global_step < self.end:
            self.last_step += 1
            values = self._get_value()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value = data
                param_group[self.param_name] = value
                self.print_value(self.verbose, i, value)

        self._last_value = [
            group[self.param_name] for group in self.optimizer.param_groups
        ]

@OPTIM_WRAPPERS.register_module()
class OptimWrapper(BaseOptimWrapper):
    def __init__(self,
                 optimizer: Optimizer,
                 accumulative_counts: int = 1,
                 clip_grad: Optional[dict] = None):
        assert accumulative_counts > 0, (
            '_accumulative_counts at least greater than or equal to 1')
        self._accumulative_counts = accumulative_counts
        self.optimizer = optimizer

        if clip_grad is not None:
            # clip_grad_kwargs should not be non-empty dict.
            assert isinstance(clip_grad, dict) and clip_grad, (
                'If `clip_grad` is not None, it should be a `dict` '
                'which is the arguments of `torch.nn.utils.clip_grad_norm_` '
                'or clip_grad_value_`.')
            clip_type = clip_grad.pop('type', 'norm')
            if clip_type == 'norm':
                self.clip_func = torch.nn.utils.clip_grad_norm_
                self.grad_name = 'grad_norm'
            elif clip_type == 'value':
                self.clip_func = torch.nn.utils.clip_grad_value_
                self.grad_name = 'grad_value'
            else:
                raise ValueError('type of clip_grad should be "norm" or '
                                 f'"value" but got {clip_type}')
            assert clip_grad, ('`clip_grad` should contain other arguments '
                               'besides `type`. The arguments should match '
                               'with the `torch.nn.utils.clip_grad_norm_` or '
                               'clip_grad_value_`')
        self.clip_grad_kwargs = clip_grad
        # Used to update `grad_norm` log message.
        self.message_hub = MessageHub.get_current_instance()
        self._inner_count = 0
        # `_max_counts` means the total number of parameter updates.  It
        # ensures that the gradient of the last few iterations will not be
        # lost when the `_max_counts` is not divisible by
        # `accumulative_counts`.
        self._max_counts = -1
        # The `_remainder_iter` is used for calculating loss factor at the
        # last few iterations. If `_max_counts` has not been initialized,
        # the loss factor will always be the same as `_accumulative_counts`.
        self._remainder_counts = -1

        # The Following code is used to initialize `base_param_settings`.
        # `base_param_settings` is used to store the parameters that are not
        # updated by the optimizer.
        # The `base_param_settings` used for tracking the base learning in the
        # optimizer. If the optimizer has multiple parameter groups, this
        # params will not be scaled by the loss factor.
        if len(optimizer.param_groups) > 1:
            self.base_param_settings = {
                'params': torch.tensor([0.0], dtype=torch.float)
            }
            self.base_param_settings.update(**self.optimizer.defaults)
        else:
            self.base_param_settings = None  # type: ignore

    def update_params(  # type: ignore
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation.

        Provide unified ``backward`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on GradScaler during backward process.

        Note:
            If subclasses inherit from ``OptimWrapper`` override
            ``backward``, ``_inner_count +=1`` must be implemented.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`.
        """
        loss.backward(**kwargs)
        self._inner_count += 1

    def zero_grad(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.zero_grad``.

        Provide unified ``zero_grad`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.zero_grad`.
        """
        self.optimizer.zero_grad(**kwargs)

    def step(self, **kwargs) -> None:
        """A wrapper of ``Optimizer.step``.

        Provide unified ``step`` interface compatible with automatic mixed
        precision training. Subclass can overload this method to implement the
        required logic. For example, ``torch.cuda.amp`` require some extra
        operation on ``GradScaler`` during step process.

        Clip grad if :attr:`clip_grad_kwargs` is not None, and then update
        parameters.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """
        if self.clip_grad_kwargs:
            self._clip_grad()
        self.optimizer.step(**kwargs)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """A Context for gradient accumulation and automatic mix precision
        training.

        If subclasses need to enable the context for mix precision training,
        e.g., ``:class:`AmpOptimWrapper``,  the corresponding context should be
        enabled in `optim_context`. Since ``OptimWrapper`` uses default fp32
        training, ``optim_context`` will only enable the context for
        blocking the unnecessary gradient synchronization during gradient
        accumulation

        If model is an instance with ``no_sync`` method (which means
        blocking the gradient synchronization) and
        ``self._accumulative_counts != 1``. The model will not automatically
        synchronize gradients if ``cur_iter`` is divisible by
        ``self._accumulative_counts``. Otherwise, this method will enable an
        empty context.

        Args:
            model (nn.Module): The training model.
        """
        # During gradient accumulation process, the gradient synchronize
        # should only happen before updating parameters.
        if not self.should_sync() and hasattr(model, 'no_sync'):
            with model.no_sync():
                yield
        else:
            yield

    def _clip_grad(self) -> None:
        """Clip the gradients of parameters."""
        params: List[torch.Tensor] = []
        for param_group in self.optimizer.param_groups:
            params.extend(param_group['params'])

        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            grad = self.clip_func(params, **self.clip_grad_kwargs)
            # `torch.nn.utils.clip_grad_value_` will return None.
            if grad is not None:
                self.message_hub.update_scalar(f'train/{self.grad_name}',
                                               float(grad))

    def initialize_count_status(self, model: nn.Module, init_counts: int,
                                max_counts: int) -> None:
        """Initialize gradient accumulation related attributes.

        ``OptimWrapper`` can be used without calling
        ``initialize_iter_status``. However, Consider the case of  ``len(
        dataloader) == 10``, and the ``accumulative_iter == 3``. Since 10 is
        not divisible by 3, the last iteration will not trigger
        ``optimizer.step()``, resulting in one less parameter updating.

        Args:
            model (nn.Module): Training model
            init_counts (int): The initial value of the inner count.
            max_counts (int): The maximum value of the inner count.
        """
        self._inner_count = init_counts
        self._max_counts = max_counts
        if self._inner_count % self._accumulative_counts != 0:
            print_log(
                'Resumed iteration number is not divisible by '
                '`_accumulative_counts` in `GradientCumulativeOptimizerHook`, '
                'which means the gradient of some iterations is lost and the '
                'result may be influenced slightly.',
                logger='current',
                level=logging.WARNING)

        if has_batch_norm(model) and self._accumulative_counts > 1:
            print_log(
                'Gradient accumulative may slightly decrease '
                'performance because the model has BatchNorm layers.',
                logger='current',
                level=logging.WARNING)
        # Remainder of `_max_counts` divided by `_accumulative_counts`
        self._remainder_counts = self._max_counts % self._accumulative_counts

    def should_update(self) -> bool:
        """Decide whether the parameters should be updated at the current
        iteration.

        Called by :meth:`update_params` and check whether the optimizer
        wrapper should update parameters at current iteration.

        Returns:
            bool: Whether to update parameters.
        """
        return (self._inner_count % self._accumulative_counts == 0
                or self._inner_count == self._max_counts)

    def should_sync(self) -> bool:
        """Decide whether the automatic gradient synchronization should be
        allowed at the current iteration.

        It takes effect when gradient accumulation is used to skip
        synchronization at the iterations where the parameter is not updated.

        Since ``should_sync`` is called by :meth:`optim_context`, and it is
        called before :meth:`backward` which means ``self._inner_count += 1``
        has not happened yet. Therefore, ``self._inner_count += 1`` should be
        performed manually here.

        Returns:
            bool: Whether to block the automatic gradient synchronization.
        """
        return ((self._inner_count + 1) % self._accumulative_counts == 0
                or (self._inner_count + 1) == self._max_counts)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Get scaled loss according to ``_accumulative_counts``,
        ``_inner_count`` and max_counts.

        Args:
            loss (torch.Tensor): Original loss calculated by model.

        Returns:
            loss (torch.Tensor): Scaled loss.
        """
        if self._accumulative_counts == 1:
            # update parameters without gradient accumulation. The gradient
            # should not be rescaled and `loss_factor=1`.
            loss_factor = 1
        elif self._max_counts == -1:
            loss_factor = self._accumulative_counts
        else:
            # if `self._accumulative_counts > 1`, the gradient needs to be
            # rescaled and accumulated. In most cases, `loss_factor` equals to
            # `self._accumulative_counts`. However, `self._max_counts` may not
            # be divisible by `self._accumulative_counts`, so the
            # `loss_scale` for the last few iterations needs to be
            # recalculated.
            if self._inner_count < self._max_counts - self._remainder_counts:
                loss_factor = self._accumulative_counts
            else:
                loss_factor = self._remainder_counts
            assert loss_factor > 0, (
                'loss_factor should be larger than zero! This error could '
                'happened when initialize_iter_status called with an '
                'error `init_counts` or `max_counts`')

        loss = loss / loss_factor
        return loss

    @property
    def inner_count(self):
        """Get the number of updating parameters of optimizer wrapper."""
        return self._inner_count

    def __repr__(self):
        wrapper_info = (f'Type: {type(self).__name__}\n'
                        f'_accumulative_counts: {self._accumulative_counts}\n'
                        'optimizer: \n')
        optimizer_str = repr(self.optimizer) + '\n'
        return wrapper_info + optimizer_str
    

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Union

from torch.utils.data import DataLoader

class OptimWrapperDict(OptimWrapper):
    def __init__(self, **optim_wrapper_dict: OptimWrapper):
        for key, value in optim_wrapper_dict.items():
            assert isinstance(value, OptimWrapper), (
                '`OptimWrapperDict` only accept OptimWrapper instance, '
                f'but got {key}: {type(value)}')
        self.optim_wrappers = optim_wrapper_dict

    def update_params(  # type: ignore
            self,
            loss: torch.Tensor,
            step_kwargs: Optional[Dict] = None,
            zero_kwargs: Optional[Dict] = None) -> None:
        """Update all optimizer wrappers would lead to a duplicate backward
        errors, and OptimWrapperDict does not know which optimizer wrapper
        should be updated.

        Therefore, this method is not implemented. The optimizer wrapper of
        OptimWrapperDict should be accessed and call its `update_params`.
        """
        raise NotImplementedError('`update_params` should be called by each '
                                  'optimizer separately`')

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Since OptimWrapperDict doesn't know which optimizer wrapper's
        backward method should be called (``loss_scaler`` maybe different in
        different :obj:AmpOptimWrapper), this method is not implemented.

        The optimizer wrapper of OptimWrapperDict should be accessed and call
        its `backward`.
        """
        raise NotImplementedError('`backward` should be called by each '
                                  'optimizer separately`')

    def step(self, **kwargs) -> None:
        """Since the backward method is not implemented, the step should not be
        implemented either."""
        raise NotImplementedError('`step` should be called by each '
                                  'optimizer separately`')

    def zero_grad(self, **kwargs) -> None:
        """Set the gradients of all optimizer wrappers to zero."""
        for optim_wrapper in self.optim_wrappers.values():
            optim_wrapper.zero_grad()

    @contextmanager
    def optim_context(self, model: nn.Module):
        """``optim_context`` should be called by each optimizer separately."""
        raise NotImplementedError(
            '`optim_context` should be called by each optimizer separately')

    def initialize_count_status(self, model: nn.Module, cur_iter,
                                max_iters) -> None:
        """Do nothing but provide unified interface for :obj:`OptimWrapper`

        Since ``OptimWrapperDict`` does not know the correspondence between
        model and optimizer wrapper. ``initialize_iter_status`` will do nothing
        and each optimizer wrapper should call ``initialize_iter_status``
        separately.
        """
        return

    @property
    def param_groups(self):
        """Returns the parameter groups of each OptimWrapper."""
        param_groups = dict()
        for key, value in self.optim_wrappers.items():
            param_groups[key] = value.param_groups
        return param_groups

    def get_lr(self) -> Dict[str, List[float]]:
        """Get the learning rate of all optimizers.

        Returns:
            Dict[str, List[float]]: Learning rate of all optimizers.
        """
        lr_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            inner_lr_dict = optim_wrapper.get_lr()
            if 'base_lr' in inner_lr_dict:
                lr_dict[f'{name}.base_lr'] = inner_lr_dict['base_lr']
            lr_dict[f'{name}.lr'] = inner_lr_dict['lr']
        return lr_dict

    def get_momentum(self) -> Dict[str, List[float]]:
        """Get the momentum of all optimizers.

        Returns:
            Dict[str, List[float]]: momentum of all optimizers.
        """
        momentum_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            momentum_dict[f'{name}.momentum'] = optim_wrapper.get_momentum(
            )['momentum']
        return momentum_dict

    def state_dict(self) -> dict:
        """Get the state dictionary of all optimizer wrappers.

        Returns:
            dict: Each key-value pair in the dictionary represents the name
            and state dictionary of corresponding :obj:`OptimWrapper`.
        """
        state_dict = dict()
        for name, optim_wrapper in self.optim_wrappers.items():
            state_dict[name] = optim_wrapper.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the state dictionary from the ``state_dict``.

        Args:
            state_dict (dict): Each key-value pair in `state_dict` represents
                the name and the state dictionary of corresponding
                :obj:`OptimWrapper`.
        """
        for name, _state_dict in state_dict.items():
            assert name in self.optim_wrappers, (
                f'Mismatched `state_dict`! cannot found {name} in '
                'OptimWrapperDict')
            self.optim_wrappers[name].load_state_dict(_state_dict)

    def items(self) -> Iterator[Tuple[str, OptimWrapper]]:
        """A generator to get the name and corresponding
        :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.items()

    def values(self) -> Iterator[OptimWrapper]:
        """A generator to get :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.values()

    def keys(self) -> Iterator[str]:
        """A generator to get the name of :obj:`OptimWrapper`"""
        yield from self.optim_wrappers.keys()

    def __getitem__(self, key: str) -> OptimWrapper:
        assert key in self.optim_wrappers, (
            f'Cannot find {key} in OptimWrapperDict, please check '
            'your optimizer constructor.')
        return self.optim_wrappers[key]

    def __contains__(self, key: str) -> bool:
        return key in self.optim_wrappers

    def __len__(self) -> int:
        return len(self.optim_wrappers)

    def __repr__(self) -> str:
        desc = ''
        for name, optim_wrapper in self.optim_wrappers.items():
            desc += f'name: {name}\n'
            desc += repr(optim_wrapper)
        return desc

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]

class BaseLoop(metaclass=ABCMeta):
    def __init__(self, runner, dataloader: Union[DataLoader, Dict]) -> None:
        self._runner = runner
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader = runner.build_dataloader(
                dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self) -> Any:
        """Execute loop."""

class _SlicedDataset:

    def __init__(self, dataset, length) -> None:
        self._dataset = dataset
        self._length = length

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return self._length

class BaseMetric(metaclass=ABCMeta):
    default_prefix: Optional[str] = None

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None) -> None:
        if collect_dir is not None and collect_device != 'cpu':
            raise ValueError('`collec_dir` could only be configured when '
                             "`collect_device='cpu'`")

        self._dataset_meta: Union[None, dict] = None
        self.collect_device = collect_device
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        self.collect_dir = collect_dir

        if self.prefix is None:
            print_log(
                'The prefix is not set in metric class '
                f'{self.__class__.__name__}.',
                logger='current',
                level=logging.WARNING)

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]
    
@EVALUATOR.register_module()
class Evaluator:
    """Wrapper class to compose multiple :class:`BaseMetric` instances.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
    """
    from custom.base import BaseDataElement

    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        self._dataset_meta: Optional[dict] = None
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
        self.metrics: List[BaseMetric] = []
        for metric in metrics:
            if isinstance(metric, dict):
                self.metrics.append(METRICS.build(metric))
            else:
                self.metrics.append(metric)

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the evaluator and it's metrics."""
        self._dataset_meta = dataset_meta
        for metric in self.metrics:
            metric.dataset_meta = dataset_meta

    def process(self,
                data_samples: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """
        _data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, BaseDataElement):
                _data_samples.append(data_sample.to_dict())
            else:
                _data_samples.append(data_sample)

        for metric in self.metrics:
            metric.process(data_batch, _data_samples)

    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """
        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)
        return metrics

    def offline_evaluate(self,
                         data_samples: Sequence,
                         data: Optional[Sequence] = None,
                         chunk_size: int = 1):
        """Offline evaluate the dumped predictions on the given data .

        Args:
            data_samples (Sequence): All predictions and ground truth of the
                model and the validation set.
            data (Sequence, optional): All data of the validation set.
            chunk_size (int): The number of data samples and predictions to be
                processed in a batch.
        """

        # support chunking iterable objects
        def get_chunks(seq: Iterator, chunk_size=1):
            stop = False
            while not stop:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(next(seq))
                    except StopIteration:
                        stop = True
                        break
                if chunk:
                    yield chunk

        if data is not None:
            assert len(data_samples) == len(data), (
                'data_samples and data should have the same length, but got '
                f'data_samples length: {len(data_samples)} '
                f'data length: {len(data)}')
            data = get_chunks(iter(data), chunk_size)

        size = 0
        for output_chunk in get_chunks(iter(data_samples), chunk_size):
            if data is not None:
                data_chunk = pseudo_collate(next(data))  # type: ignore
            else:
                data_chunk = None
            size += len(output_chunk)
            self.process(output_chunk, data_chunk)
        return self.evaluate(size)

class Hook:
    """Base hook class.

    All hooks should inherit from this class.
    """

    priority = 'NORMAL'
    stages = ('before_run', 'after_load_checkpoint', 'before_train',
              'before_train_epoch', 'before_train_iter', 'after_train_iter',
              'after_train_epoch', 'before_val', 'before_val_epoch',
              'before_val_iter', 'after_val_iter', 'after_val_epoch',
              'after_val', 'before_save_checkpoint', 'after_train',
              'before_test', 'before_test_epoch', 'before_test_iter',
              'after_test_iter', 'after_test_epoch', 'after_test', 'after_run')

    def before_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before the training validation or testing process.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
        """

    def after_run(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before the training validation or testing process.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
        """

    def before_train(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before train.

        Args:
            runner (Runner): The runner of the training process.
        """

    def after_train(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after train.

        Args:
            runner (Runner): The runner of the training process.
        """

    def before_val(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before validation.

        Args:
            runner (Runner): The runner of the validation process.
        """

    def after_val(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after validation.

        Args:
            runner (Runner): The runner of the validation process.
        """

    def before_test(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before testing.

        Args:
            runner (Runner): The runner of the testing process.
        """

    def after_test(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after testing.

        Args:
            runner (Runner): The runner of the testing process.
        """

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations before saving the checkpoint.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """All subclasses should override this method, if they need any
        operations after loading the checkpoint.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.
        """

    def before_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._before_epoch(runner, mode='train')

    def before_val_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
        """
        self._before_epoch(runner, mode='val')

    def before_test_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
        """
        self._before_epoch(runner, mode='test')

    def after_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        self._after_epoch(runner, mode='train')

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._after_epoch(runner, mode='val')

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test epoch.

        Args:
            runner (Runner): The runner of the testing process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on test dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        self._after_epoch(runner, mode='test')

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='train')

    def before_val_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='val')

    def before_test_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None) -> None:
        """All subclasses should override this method, if they need any
        operations before each test iteration.

        Args:
            runner (Runner): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
                Defaults to None.
        """
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='test')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='train')

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation iteration.

        Args:
            runner (Runner): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='val')

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each test iteration.

        Args:
            runner (Runner): The runner of the training  process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='test')

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations before each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """

    def _after_epoch(self, runner, mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            mode (str): Current mode of runner. Defaults to 'train'.
        """

    def _before_iter(self,
                     runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None,
                     mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations before each iter.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            mode (str): Current mode of runner. Defaults to 'train'.
        """

    def _after_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Union[Sequence, dict]] = None,
                    mode: str = 'train') -> None:
        """All subclasses should override this method, if they need any
        operations after each epoch.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            batch_idx (int): The index of the current batch in the loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict or Sequence, optional): Outputs from model.
            mode (str): Current mode of runner. Defaults to 'train'.
        """

    def every_n_epochs(self, runner, n: int, start: int = 0) -> bool:
        """Test whether current epoch can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            n (int): Whether current epoch can be evenly divided by n.
            start (int): Starting from `start` to check the logic for
                every n epochs. Defaults to 0.

        Returns:
            bool: Whether current epoch can be evenly divided by n.
        """
        dividend = runner.epoch + 1 - start
        return dividend % n == 0 if dividend >= 0 and n > 0 else False

    def every_n_inner_iters(self, batch_idx: int, n: int) -> bool:
        """Test whether current inner iteration can be evenly divided by n.

        Args:
            batch_idx (int): Current batch index of the training, validation
                or testing loop.
            n (int): Whether current inner iteration can be evenly
                divided by n.

        Returns:
            bool: Whether current inner iteration can be evenly
            divided by n.
        """
        return (batch_idx + 1) % n == 0 if n > 0 else False

    def every_n_train_iters(self, runner, n: int, start: int = 0) -> bool:
        """Test whether current training iteration can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            n (int): Whether current iteration can be evenly divided by n.
            start (int): Starting from `start` to check the logic for
                every n iterations. Defaults to 0.

        Returns:
            bool: Return True if the current iteration can be evenly divided
            by n, otherwise False.
        """
        dividend = runner.iter + 1 - start
        return dividend % n == 0 if dividend >= 0 and n > 0 else False

    def end_of_epoch(self, dataloader, batch_idx: int) -> bool:
        """Check whether the current iteration reaches the last iteration of
        the dataloader.

        Args:
            dataloader (Dataloader): The dataloader of the training,
                validation or testing process.
            batch_idx (int): The index of the current batch in the loop.
        Returns:
            bool: Whether reaches the end of current epoch or not.
        """
        return batch_idx + 1 == len(dataloader)

    def is_last_train_epoch(self, runner) -> bool:
        """Test whether current epoch is the last train epoch.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: Whether reaches the end of training epoch.
        """
        return runner.epoch + 1 == runner.max_epochs

    def is_last_train_iter(self, runner) -> bool:
        """Test whether current iteration is the last train iteration.

        Args:
            runner (Runner): The runner of the training process.

        Returns:
            bool: Whether current iteration is the last train iteration.
        """
        return runner.iter + 1 == runner.max_iters

    def get_triggered_stages(self) -> list:
        """Get all triggered stages with method name of the hook.

        Returns:
            list: List of triggered stages.
        """
        trigger_stages = set()
        for stage in Hook.stages:
            if is_method_overridden(stage, Hook, self):
                trigger_stages.add(stage)

        # some methods will be triggered in multi stages
        # use this dict to map method to stages.
        method_stages_map = {
            '_before_epoch':
            ['before_train_epoch', 'before_val_epoch', 'before_test_epoch'],
            '_after_epoch':
            ['after_train_epoch', 'after_val_epoch', 'after_test_epoch'],
            '_before_iter':
            ['before_train_iter', 'before_val_iter', 'before_test_iter'],
            '_after_iter':
            ['after_train_iter', 'after_val_iter', 'after_test_iter'],
        }

        for method, map_stages in method_stages_map.items():
            if is_method_overridden(method, Hook, self):
                trigger_stages.update(map_stages)

        return list(trigger_stages)

class ManagerMeta(type):
    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert 'name' in params_names, f'{cls} must have the `name` argument'
        super().__init__(*args)


class ManagerMixin(metaclass=ManagerMeta):
    """``ManagerMixin`` is the base class for classes that have global access
    requirements.

    The subclasses inheriting from ``ManagerMixin`` can get their
    global instances.

    Examples:
        >>> class GlobalAccessible(ManagerMixin):
        >>>     def __init__(self, name=''):
        >>>         super().__init__(name)
        >>>
        >>> GlobalAccessible.get_instance('name')
        >>> instance_1 = GlobalAccessible.get_instance('name')
        >>> instance_2 = GlobalAccessible.get_instance('name')
        >>> assert id(instance_1) == id(instance_2)

    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: str = '', **kwargs):
        assert isinstance(name, str) and name, \
            'name argument must be an non-empty string.'
        self._instance_name = name

    @classmethod
    def get_instance(cls: Type[T], name: str, **kwargs) -> T:
        """Get subclass instance by name if the name exists.

        If corresponding name instance has not been created, ``get_instance``
        will create an instance, otherwise ``get_instance`` will return the
        corresponding instance.

        Examples
            >>> instance1 = GlobalAccessible.get_instance('name1')
            >>> # Create name1 instance.
            >>> instance.instance_name
            name1
            >>> instance2 = GlobalAccessible.get_instance('name1')
            >>> # Get name1 instance.
            >>> assert id(instance1) == id(instance2)

        Args:
            name (str): Name of instance. Defaults to ''.

        Returns:
            object: Corresponding name instance, the latest instance, or root
            instance.
        """
        _accquire_lock()
        assert isinstance(name, str), \
            f'type of name should be str, but got {type(cls)}'
        instance_dict = cls._instance_dict  # type: ignore
        # Get the instance by name.
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)  # type: ignore
            instance_dict[name] = instance  # type: ignore
        elif kwargs:
            warnings.warn(
                f'{cls} instance named of {name} has been created, '
                'the method `get_instance` should not accept any other '
                'arguments')
        # Get latest instantiated instance or root instance.
        _release_lock()
        return instance_dict[name]

    @classmethod
    def get_current_instance(cls):
        """Get latest created instance.

        Before calling ``get_current_instance``, The subclass must have called
        ``get_instance(xxx)`` at least once.

        Examples
            >>> instance = GlobalAccessible.get_current_instance()
            AssertionError: At least one of name and current needs to be set
            >>> instance = GlobalAccessible.get_instance('name1')
            >>> instance.instance_name
            name1
            >>> instance = GlobalAccessible.get_current_instance()
            >>> instance.instance_name
            name1

        Returns:
            object: Latest created instance.
        """
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(
                f'Before calling {cls.__name__}.get_current_instance(), you '
                'should call get_instance(name=xxx) at least once.')
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        """Check whether the name corresponding instance exists.

        Args:
            name (str): Name of instance.

        Returns:
            bool: Whether the name corresponding instance exists.
        """
        return name in cls._instance_dict

    @property
    def instance_name(self) -> str:
        """Get the name of instance.

        Returns:
            str: Name of instance.
        """
        return self._instance_name
class HistoryBuffer:
    _statistics_methods: dict = dict()

    def __init__(self,
                 log_history: Sequence = [],
                 count_history: Sequence = [],
                 max_length: int = 1000000):

        self.max_length = max_length
        self._set_default_statistics()
        assert len(log_history) == len(count_history), \
            'The lengths of log_history and count_histroy should be equal'
        if len(log_history) > max_length:
            warnings.warn(f'The length of history buffer({len(log_history)}) '
                          f'exceeds the max_length({max_length}), the first '
                          'few elements will be ignored.')
            self._log_history = np.array(log_history[-max_length:])
            self._count_history = np.array(count_history[-max_length:])
        else:
            self._log_history = np.array(log_history)
            self._count_history = np.array(count_history)

    def _set_default_statistics(self) -> None:
        """Register default statistic methods: min, max, current and mean."""
        self._statistics_methods.setdefault('min', HistoryBuffer.min)
        self._statistics_methods.setdefault('max', HistoryBuffer.max)
        self._statistics_methods.setdefault('current', HistoryBuffer.current)
        self._statistics_methods.setdefault('mean', HistoryBuffer.mean)

    def update(self, log_val: Union[int, float], count: int = 1) -> None:
        """update the log history.

        If the length of the buffer exceeds ``self._max_length``, the oldest
        element will be removed from the buffer.

        Args:
            log_val (int or float): The value of log.
            count (int): The accumulation times of log, defaults to 1.
            ``count`` will be used in smooth statistics.
        """
        if (not isinstance(log_val, (int, float))
                or not isinstance(count, (int, float))):
            raise TypeError(f'log_val must be int or float but got '
                            f'{type(log_val)}, count must be int but got '
                            f'{type(count)}')
        self._log_history = np.append(self._log_history, log_val)
        self._count_history = np.append(self._count_history, count)
        if len(self._log_history) > self.max_length:
            self._log_history = self._log_history[-self.max_length:]
            self._count_history = self._count_history[-self.max_length:]

    @property
    def data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the ``_log_history`` and ``_count_history``.

        Returns:
            Tuple[np.ndarray, np.ndarray]: History logs and the counts of
            the history logs.
        """
        return self._log_history, self._count_history

    @classmethod
    def register_statistics(cls, method: Callable) -> Callable:
        """Register custom statistics method to ``_statistics_methods``.

        The registered method can be called by ``history_buffer.statistics``
        with corresponding method name and arguments.

        Examples:
            >>> @HistoryBuffer.register_statistics
            >>> def weighted_mean(self, window_size, weight):
            >>>     assert len(weight) == window_size
            >>>     return (self._log_history[-window_size:] *
            >>>             np.array(weight)).sum() / \
            >>>             self._count_history[-window_size:]

            >>> log_buffer = HistoryBuffer([1, 2], [1, 1])
            >>> log_buffer.statistics('weighted_mean', 2, [2, 1])
            2

        Args:
            method (Callable): Custom statistics method.
        Returns:
            Callable: Original custom statistics method.
        """
        method_name = method.__name__
        assert method_name not in cls._statistics_methods, \
            'method_name cannot be registered twice!'
        cls._statistics_methods[method_name] = method
        return method

    def statistics(self, method_name: str, *arg, **kwargs) -> Any:
        """Access statistics method by name.

        Args:
            method_name (str): Name of method.

        Returns:
            Any: Depends on corresponding method.
        """
        if method_name not in self._statistics_methods:
            raise KeyError(f'{method_name} has not been registered in '
                           'HistoryBuffer._statistics_methods')
        method = self._statistics_methods[method_name]
        # Provide self arguments for registered functions.
        return method(self, *arg, **kwargs)

    def mean(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the mean of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global mean value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: Mean value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        logs_sum = self._log_history[-window_size:].sum()
        counts_sum = self._count_history[-window_size:].sum()
        return logs_sum / counts_sum

    def max(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the maximum value of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global maximum value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: The maximum value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].max()

    def min(self, window_size: Optional[int] = None) -> np.ndarray:
        """Return the minimum value of the latest ``window_size`` values in log
        histories.

        If ``window_size is None`` or ``window_size > len(self._log_history)``,
        return the global minimum value of history logs.

        Args:
            window_size (int, optional): Size of statistics window.
        Returns:
            np.ndarray: The minimum value within the window.
        """
        if window_size is not None:
            assert isinstance(window_size, int), \
                'The type of window size should be int, but got ' \
                f'{type(window_size)}'
        else:
            window_size = len(self._log_history)
        return self._log_history[-window_size:].min()

    def current(self) -> np.ndarray:
        """Return the recently updated values in log histories.

        Returns:
            np.ndarray: Recently updated values in log histories.
        """
        if len(self._log_history) == 0:
            raise ValueError('HistoryBuffer._log_history is an empty array! '
                             'please call update first')
        return self._log_history[-1]

    def __getstate__(self) -> dict:
        """Make ``_statistics_methods`` can be resumed.

        Returns:
            dict: State dict including statistics_methods.
        """
        self.__dict__.update(statistics_methods=self._statistics_methods)
        return self.__dict__

    def __setstate__(self, state):
        """Try to load ``_statistics_methods`` from state.

        Args:
            state (dict): State dict.
        """
        statistics_methods = state.pop('statistics_methods', {})
        self._set_default_statistics()
        self._statistics_methods.update(statistics_methods)
        self.__dict__.update(state)
        
class MessageHub(ManagerMixin):
    def __init__(self,
                 name: str,
                 log_scalars: Optional[dict] = None,
                 runtime_info: Optional[dict] = None,
                 resumed_keys: Optional[dict] = None):
        super().__init__(name)
        self._log_scalars = self._parse_input('log_scalars', log_scalars)
        self._runtime_info = self._parse_input('runtime_info', runtime_info)
        self._resumed_keys = self._parse_input('resumed_keys', resumed_keys)

        for value in self._log_scalars.values():
            assert isinstance(value, HistoryBuffer), \
                ("The type of log_scalars'value must be HistoryBuffer, but "
                 f'got {type(value)}')

        for key in self._resumed_keys.keys():
            assert key in self._log_scalars or key in self._runtime_info, \
                ('Key in `resumed_keys` must contained in `log_scalars` or '
                 f'`runtime_info`, but got {key}')

    @classmethod
    def get_current_instance(cls) -> 'MessageHub':
        """Get latest created ``MessageHub`` instance.

        :obj:`MessageHub` can call :meth:`get_current_instance` before any
        instance has been created, and return a message hub with the instance
        name "mmengine".

        Returns:
            MessageHub: Empty ``MessageHub`` instance.
        """
        if not cls._instance_dict:
            cls.get_instance('mmengine')
        return super().get_current_instance()

    def update_scalar(self,
                      key: str,
                      value: Union[int, float, np.ndarray, 'torch.Tensor'],
                      count: int = 1,
                      resumed: bool = True) -> None:
        self._set_resumed_keys(key, resumed)
        checked_value = self._get_valid_value(value)
        assert isinstance(count, int), (
            f'The type of count must be int. but got {type(count): {count}}')
        if key in self._log_scalars:
            self._log_scalars[key].update(checked_value, count)
        else:
            self._log_scalars[key] = HistoryBuffer([checked_value], [count])

    def update_scalars(self, log_dict: dict, resumed: bool = True) -> None:
        """Update :attr:`_log_scalars` with a dict.

        ``update_scalars`` iterates through each pair of log_dict key-value,
        and calls ``update_scalar``. If type of value is dict, the value should
        be ``dict(value=xxx) or dict(value=xxx, count=xxx)``. Item in
        ``log_dict`` has the same resume option.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``log_dict``.

        Args:
            log_dict (str): Used for batch updating :attr:`_log_scalars`.
            resumed (bool): Whether all ``HistoryBuffer`` referred in
                log_dict should be resumed. Defaults to True.

        Examples:
            >>> message_hub = MessageHub.get_instance('mmengine')
            >>> log_dict = dict(a=1, b=2, c=3)
            >>> message_hub.update_scalars(log_dict)
            >>> # The default count of  `a`, `b` and `c` is 1.
            >>> log_dict = dict(a=1, b=2, c=dict(value=1, count=2))
            >>> message_hub.update_scalars(log_dict)
            >>> # The count of `c` is 2.
        """
        assert isinstance(log_dict, dict), ('`log_dict` must be a dict!, '
                                            f'but got {type(log_dict)}')
        for log_name, log_val in log_dict.items():
            if isinstance(log_val, dict):
                assert 'value' in log_val, \
                    f'value must be defined in {log_val}'
                count = self._get_valid_value(log_val.get('count', 1))
                value = log_val['value']
            else:
                count = 1
                value = log_val
            assert isinstance(count,
                              int), ('The type of count must be int. but got '
                                     f'{type(count): {count}}')
            self.update_scalar(log_name, value, count, resumed)

    def update_info(self, key: str, value: Any, resumed: bool = True) -> None:
        """Update runtime information.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``key``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> message_hub.update_info('iter', 100)

        Args:
            key (str): Key of runtime information.
            value (Any): Value of runtime information.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        self._set_resumed_keys(key, resumed)
        self._runtime_info[key] = value

    def pop_info(self, key: str, default: Optional[Any] = None) -> Any:
        """Remove runtime information by key. If the key does not exist, this
        method will return the default value.

        Args:
            key (str): Key of runtime information.
            default (Any, optional): The default returned value for the
                given key.

        Returns:
            Any: The runtime information if the key exists.
        """
        return self._runtime_info.pop(key, default)

    def update_info_dict(self, info_dict: dict, resumed: bool = True) -> None:
        """Update runtime information with dictionary.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            The ``resumed`` argument needs to be consistent for the same
            ``info_dict``.

        Examples:
            >>> message_hub = MessageHub(name='name')
            >>> message_hub.update_info({'iter': 100})

        Args:
            info_dict (str): Runtime information dictionary.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        assert isinstance(info_dict, dict), ('`log_dict` must be a dict!, '
                                             f'but got {type(info_dict)}')
        for key, value in info_dict.items():
            self.update_info(key, value, resumed=resumed)

    def _set_resumed_keys(self, key: str, resumed: bool) -> None:
        """Set corresponding resumed keys.

        This method is called by ``update_scalar``, ``update_scalars`` and
        ``update_info`` to set the corresponding key is true or false in
        :attr:`_resumed_keys`.

        Args:
            key (str): Key of :attr:`_log_scalrs` or :attr:`_runtime_info`.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        if key not in self._resumed_keys:
            self._resumed_keys[key] = resumed
        else:
            assert self._resumed_keys[key] == resumed, \
                f'{key} used to be {self._resumed_keys[key]}, but got ' \
                '{resumed} now. resumed keys cannot be modified repeatedly.'

    @property
    def log_scalars(self) -> OrderedDict:
        """Get all ``HistoryBuffer`` instances.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will return a reference of
            history buffer rather than a copy.

        Returns:
            OrderedDict: All ``HistoryBuffer`` instances.
        """
        return self._log_scalars

    @property
    def runtime_info(self) -> OrderedDict:
        """Get all runtime information.

        Returns:
            OrderedDict: A copy of all runtime information.
        """
        return self._runtime_info

    def get_scalar(self, key: str) -> HistoryBuffer:
        """Get ``HistoryBuffer`` instance by key.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will not return a reference of
            history buffer rather than a copy.

        Args:
            key (str): Key of ``HistoryBuffer``.

        Returns:
            HistoryBuffer: Corresponding ``HistoryBuffer`` instance if the
            key exists.
        """
        if key not in self.log_scalars:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self.log_scalars[key]

    def get_info(self, key: str, default: Optional[Any] = None) -> Any:
        """Get runtime information by key. If the key does not exist, this
        method will return default information.

        Args:
            key (str): Key of runtime information.
            default (Any, optional): The default returned value for the
                given key.

        Returns:
            Any: A copy of corresponding runtime information if the key exists.
        """
        if key not in self.runtime_info:
            return default
        else:
            # TODO: There are restrictions on objects that can be saved
            # return copy.deepcopy(self._runtime_info[key])
            return self._runtime_info[key]

    def _get_valid_value(
        self,
        value: Union['torch.Tensor', np.ndarray, np.number, int, float],
    ) -> Union[int, float]:
        """Convert value to python built-in type.

        Args:
            value (torch.Tensor or np.ndarray or np.number or int or float):
                value of log.

        Returns:
            float or int: python built-in type value.
        """
        if isinstance(value, (np.ndarray, np.number)):
            assert value.size == 1
            value = value.item()
        elif isinstance(value, (int, float)):
            value = value
        else:
            # check whether value is torch.Tensor but don't want
            # to import torch in this file
            assert hasattr(value, 'numel') and value.numel() == 1
            value = value.item()
        return value  # type: ignore

    def state_dict(self) -> dict:
        """Returns a dictionary containing log scalars, runtime information and
        resumed keys, which should be resumed.

        The returned ``state_dict`` can be loaded by :meth:`load_state_dict`.

        Returns:
            dict: A dictionary contains ``log_scalars``, ``runtime_info`` and
            ``resumed_keys``.
        """
        saved_scalars = OrderedDict()
        saved_info = OrderedDict()

        for key, value in self._log_scalars.items():
            if self._resumed_keys.get(key, False):
                saved_scalars[key] = copy.deepcopy(value)

        for key, value in self._runtime_info.items():
            if self._resumed_keys.get(key, False):
                try:
                    saved_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    print_log(
                        f'{key} in message_hub cannot be copied, '
                        f'just return its reference. ',
                        logger='current',
                        level=logging.WARNING)
                    saved_info[key] = value
        return dict(
            log_scalars=saved_scalars,
            runtime_info=saved_info,
            resumed_keys=self._resumed_keys)

    def load_state_dict(self, state_dict: Union['MessageHub', dict]) -> None:
        """Loads log scalars, runtime information and resumed keys from
        ``state_dict`` or ``message_hub``.

        If ``state_dict`` is a dictionary returned by :meth:`state_dict`, it
        will only make copies of data which should be resumed from the source
        ``message_hub``.

        If ``state_dict`` is a ``message_hub`` instance, it will make copies of
        all data from the source message_hub. We suggest to load data from
        ``dict`` rather than a ``MessageHub`` instance.

        Args:
            state_dict (dict or MessageHub): A dictionary contains key
                ``log_scalars`` ``runtime_info`` and ``resumed_keys``, or a
                MessageHub instance.
        """
        if isinstance(state_dict, dict):
            for key in ('log_scalars', 'runtime_info', 'resumed_keys'):
                assert key in state_dict, (
                    'The loaded `state_dict` of `MessageHub` must contain '
                    f'key: `{key}`')
            # The old `MessageHub` could save non-HistoryBuffer `log_scalars`,
            # therefore the loaded `log_scalars` needs to be filtered.
            for key, value in state_dict['log_scalars'].items():
                if not isinstance(value, HistoryBuffer):
                    print_log(
                        f'{key} in message_hub is not HistoryBuffer, '
                        f'just skip resuming it.',
                        logger='current',
                        level=logging.WARNING)
                    continue
                self.log_scalars[key] = value

            for key, value in state_dict['runtime_info'].items():
                try:
                    self._runtime_info[key] = copy.deepcopy(value)
                except:  # noqa: E722
                    print_log(
                        f'{key} in message_hub cannot be copied, '
                        f'just return its reference.',
                        logger='current',
                        level=logging.WARNING)
                    self._runtime_info[key] = value

            for key, value in state_dict['resumed_keys'].items():
                if key not in set(self.log_scalars.keys()) | \
                        set(self._runtime_info.keys()):
                    print_log(
                        f'resumed key: {key} is not defined in message_hub, '
                        f'just skip resuming this key.',
                        logger='current',
                        level=logging.WARNING)
                    continue
                elif not value:
                    print_log(
                        f'Although resumed key: {key} is False, {key} '
                        'will still be loaded this time. This key will '
                        'not be saved by the next calling of '
                        '`MessageHub.state_dict()`',
                        logger='current',
                        level=logging.WARNING)
                self._resumed_keys[key] = value

        # Since some checkpoints saved serialized `message_hub` instance,
        # `load_state_dict` support loading `message_hub` instance for
        # compatibility
        else:
            self._log_scalars = copy.deepcopy(state_dict._log_scalars)
            self._runtime_info = copy.deepcopy(state_dict._runtime_info)
            self._resumed_keys = copy.deepcopy(state_dict._resumed_keys)

    def _parse_input(self, name: str, value: Any) -> OrderedDict:
        """Parse input value.

        Args:
            name (str): name of input value.
            value (Any): Input value.

        Returns:
            dict: Parsed input value.
        """
        if value is None:
            return OrderedDict()
        elif isinstance(value, dict):
            return OrderedDict(value)
        else:
            raise TypeError(f'{name} should be a dict or `None`, but '
                            f'got {type(name)}')
        
@VISUALIZERS.register_module()
class Visualizer(ManagerMixin):
    def __init__(
        self,
        name='visualizer',
        image: Optional[np.ndarray] = None,
        vis_backends: VisBackendsType = None,
        save_dir: Optional[str] = None,
        fig_save_cfg=dict(frameon=False),
        fig_show_cfg=dict(frameon=False)
    ) -> None:
        super().__init__(name)
        self._dataset_meta: Optional[dict] = None
        self._vis_backends: Dict[str, BaseVisBackend] = {}

        if vis_backends is None:
            vis_backends = []

        if isinstance(vis_backends, (dict, BaseVisBackend)):
            vis_backends = [vis_backends]  # type: ignore

        if not is_seq_of(vis_backends, (dict, BaseVisBackend)):
            raise TypeError('vis_backends must be a list of dicts or a list '
                            'of BaseBackend instances')
        if save_dir is not None:
            save_dir = osp.join(save_dir, 'vis_data')

        for vis_backend in vis_backends:  # type: ignore
            name = None
            if isinstance(vis_backend, dict):
                name = vis_backend.pop('name', None)
                vis_backend.setdefault('save_dir', save_dir)
                vis_backend = VISBACKENDS.build(vis_backend)

            # If vis_backend requires `save_dir` (with no default value)
            # but is initialized with None, then don't add this
            # vis_backend to the visualizer.
            save_dir_arg = inspect.signature(
                vis_backend.__class__.__init__).parameters.get('save_dir')
            if (save_dir_arg is not None
                    and save_dir_arg.default is save_dir_arg.empty
                    and getattr(vis_backend, '_save_dir') is None):
                warnings.warn(f'Failed to add {vis_backend.__class__}, '
                              'please provide the `save_dir` argument.')
                continue

            type_name = vis_backend.__class__.__name__
            name = name or type_name

            if name in self._vis_backends:
                raise RuntimeError(f'vis_backend name {name} already exists')
            self._vis_backends[name] = vis_backend  # type: ignore

        self.fig_save = None
        self.fig_save_cfg = fig_save_cfg
        self.fig_show_cfg = fig_show_cfg

        (self.fig_save_canvas, self.fig_save,
         self.ax_save) = self._initialize_fig(fig_save_cfg)
        self.dpi = self.fig_save.get_dpi()

        if image is not None:
            self.set_image(image)

    @property  # type: ignore
    @master_only
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter  # type: ignore
    @master_only
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the Visualizer."""
        self._dataset_meta = dataset_meta

    @master_only
    def show(self,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: float = 0.,
             continue_key: str = ' ',
             backend: str = 'matplotlib') -> None:
        """Show the drawn image.

        Args:
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (float): Delay in seconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
            backend (str): The backend to show the image. Defaults to
                'matplotlib'. `New in version 0.7.3.`
        """
        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            is_inline = 'inline' in plt.get_backend()
            img = self.get_image() if drawn_img is None else drawn_img
            self._init_manager(win_name)
            fig = self.manager.canvas.figure
            # remove white edges by set subplot margin
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.clear()
            ax = fig.add_subplot()
            ax.axis(False)
            ax.imshow(img)
            self.manager.canvas.draw()

            # Find a better way for inline to show the image
            if is_inline:
                return fig
            wait_continue(fig, timeout=wait_time, continue_key=continue_key)
        elif backend == 'cv2':
            # Keep images are shown in the same window, and the title of window
            # will be updated with `win_name`.
            cv2.namedWindow(winname=f'{id(self)}')
            cv2.setWindowTitle(f'{id(self)}', win_name)
            cv2.imshow(
                str(id(self)),
                self.get_image() if drawn_img is None else drawn_img)
            cv2.waitKey(int(np.ceil(wait_time * 1000)))
        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    @master_only
    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The image to draw.
        """
        assert image is not None
        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        # add a small 1e-2 to avoid precision lost due to matplotlib's
        # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
        self.fig_save.set_size_inches(  # type: ignore
            (self.width + 1e-2) / self.dpi, (self.height + 1e-2) / self.dpi)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')

    @master_only
    def get_image(self) -> np.ndarray:
        """Get the drawn image. The format is RGB.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        assert self._image is not None, 'Please set image using `set_image`'
        return img_from_canvas(self.fig_save_canvas)  # type: ignore

    def _initialize_fig(self, fig_cfg) -> tuple:
        """Build figure according to fig_cfg.

        Args:
            fig_cfg (dict): The config to build figure.

        Returns:
             tuple: build canvas figure and axes.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(**fig_cfg)
        ax = fig.add_subplot()
        ax.axis(False)

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvasAgg(fig)
        return canvas, fig, ax

    def _init_manager(self, win_name: str) -> None:
        """Initialize the matplot manager.

        Args:
            win_name (str): The window name.
        """
        from matplotlib.figure import Figure
        from matplotlib.pyplot import new_figure_manager
        if getattr(self, 'manager', None) is None:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)

        try:
            self.manager.set_window_title(win_name)
        except Exception:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)
            self.manager.set_window_title(win_name)

    @master_only
    def get_backend(self, name) -> 'BaseVisBackend':
        """get vis backend by name.

        Args:
            name (str): The name of vis backend

        Returns:
             BaseVisBackend: The vis backend.
        """
        return self._vis_backends.get(name)  # type: ignore

    def _is_posion_valid(self, position: np.ndarray) -> bool:
        """Judge whether the position is in image.

        Args:
            position (np.ndarray): The position to judge which last dim must
                be two and the format is [x, y].

        Returns:
            bool: Whether the position is in image.
        """
        flag = (position[..., 0] < self.width).all() and \
               (position[..., 0] >= 0).all() and \
               (position[..., 1] < self.height).all() and \
               (position[..., 1] >= 0).all()
        return flag

    @master_only
    def draw_points(self,
                    positions: Union[np.ndarray, torch.Tensor],
                    colors: Union[str, tuple, List[str], List[tuple]] = 'g',
                    marker: Optional[str] = None,
                    sizes: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """Draw single or multiple points.

        Args:
            positions (Union[np.ndarray, torch.Tensor]): Positions to draw.
            colors (Union[str, tuple, List[str], List[tuple]]): The colors
                of points. ``colors`` can have the same length with points or
                just single value. If ``colors`` is single value, all the
                points will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g.
            marker (str, optional): The marker style.
                See :mod:`matplotlib.markers` for more information about
                marker styles. Defaults to None.
            sizes (Optional[Union[np.ndarray, torch.Tensor]]): The marker size.
                Defaults to None.
        """
        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)

        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape[-1] == 2, (
            'The shape of `positions` should be (N, 2), '
            f'but got {positions.shape}')
        colors = color_val_matplotlib(colors)  # type: ignore
        self.ax_save.scatter(
            positions[:, 0], positions[:, 1], c=colors, s=sizes, marker=marker)
        return self

    @master_only
    def draw_texts(
        self,
        texts: Union[str, List[str]],
        positions: Union[np.ndarray, torch.Tensor],
        font_sizes: Optional[Union[int, List[int]]] = None,
        colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        vertical_alignments: Union[str, List[str]] = 'top',
        horizontal_alignments: Union[str, List[str]] = 'left',
        font_families: Union[str, List[str]] = 'sans-serif',
        bboxes: Optional[Union[dict, List[dict]]] = None,
        font_properties: Optional[Union['FontProperties',
                                        List['FontProperties']]] = None
    ) -> 'Visualizer':
        from matplotlib.font_manager import FontProperties
        check_type('texts', texts, (str, list))
        if isinstance(texts, str):
            texts = [texts]
        num_text = len(texts)
        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)
        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape == (num_text, 2), (
            '`positions` should have the shape of '
            f'({num_text}, 2), but got {positions.shape}')
        if not self._is_posion_valid(positions):
            warnings.warn(
                'Warning: The text is out of bounds,'
                ' the drawn text may not be in the image', UserWarning)
        positions = positions.tolist()

        if font_sizes is None:
            font_sizes = self._default_font_size
        check_type_and_length('font_sizes', font_sizes, (int, float, list),
                              num_text)
        font_sizes = value2list(font_sizes, (int, float), num_text)

        check_type_and_length('colors', colors, (str, tuple, list), num_text)
        colors = value2list(colors, (str, tuple), num_text)
        colors = color_val_matplotlib(colors)  # type: ignore

        check_type_and_length('vertical_alignments', vertical_alignments,
                              (str, list), num_text)
        vertical_alignments = value2list(vertical_alignments, str, num_text)

        check_type_and_length('horizontal_alignments', horizontal_alignments,
                              (str, list), num_text)
        horizontal_alignments = value2list(horizontal_alignments, str,
                                           num_text)

        check_type_and_length('font_families', font_families, (str, list),
                              num_text)
        font_families = value2list(font_families, str, num_text)

        if font_properties is None:
            font_properties = [None for _ in range(num_text)]  # type: ignore
        else:
            check_type_and_length('font_properties', font_properties,
                                  (FontProperties, list), num_text)
            font_properties = value2list(font_properties, FontProperties,
                                         num_text)

        if bboxes is None:
            bboxes = [None for _ in range(num_text)]  # type: ignore
        else:
            check_type_and_length('bboxes', bboxes, (dict, list), num_text)
            bboxes = value2list(bboxes, dict, num_text)

        for i in range(num_text):
            self.ax_save.text(
                positions[i][0],
                positions[i][1],
                texts[i],
                size=font_sizes[i],  # type: ignore
                bbox=bboxes[i],  # type: ignore
                verticalalignment=vertical_alignments[i],
                horizontalalignment=horizontal_alignments[i],
                family=font_families[i],
                fontproperties=font_properties[i],
                color=colors[i])
        return self

    @master_only
    def draw_lines(
        self,
        x_datas: Union[np.ndarray, torch.Tensor],
        y_datas: Union[np.ndarray, torch.Tensor],
        colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2
    ) -> 'Visualizer':
        """Draw single or multiple line segments.

        Args:
            x_datas (Union[np.ndarray, torch.Tensor]): The x coordinate of
                each line' start and end points.
            y_datas (Union[np.ndarray, torch.Tensor]): The y coordinate of
                each line' start and end points.
            colors (Union[str, tuple, List[str], List[tuple]]): The colors of
                lines. ``colors`` can have the same length with lines or just
                single value. If ``colors`` is single value, all the lines
                will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
        """
        from matplotlib.collections import LineCollection
        check_type('x_datas', x_datas, (np.ndarray, torch.Tensor))
        x_datas = tensor2ndarray(x_datas)
        check_type('y_datas', y_datas, (np.ndarray, torch.Tensor))
        y_datas = tensor2ndarray(y_datas)
        assert x_datas.shape == y_datas.shape, (
            '`x_datas` and `y_datas` should have the same shape')
        assert x_datas.shape[-1] == 2, (
            f'The shape of `x_datas` should be (N, 2), but got {x_datas.shape}'
        )
        if len(x_datas.shape) == 1:
            x_datas = x_datas[None]
            y_datas = y_datas[None]
        colors = color_val_matplotlib(colors)  # type: ignore
        lines = np.concatenate(
            (x_datas.reshape(-1, 2, 1), y_datas.reshape(-1, 2, 1)), axis=-1)
        if not self._is_posion_valid(lines):
            warnings.warn(
                'Warning: The line is out of bounds,'
                ' the drawn line may not be in the image', UserWarning)
        line_collect = LineCollection(
            lines.tolist(),
            colors=colors,
            linestyles=line_styles,
            linewidths=line_widths)
        self.ax_save.add_collection(line_collect)
        return self

    @master_only
    def draw_circles(
        self,
        center: Union[np.ndarray, torch.Tensor],
        radius: Union[np.ndarray, torch.Tensor],
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[float, int] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple circles.

        Args:
            center (Union[np.ndarray, torch.Tensor]): The x coordinate of
                each line' start and end points.
            radius (Union[np.ndarray, torch.Tensor]): The y coordinate of
                each line' start and end points.
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of circles. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value,
                all the lines will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of circles.
                Defaults to 0.8.
        """
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Circle
        check_type('center', center, (np.ndarray, torch.Tensor))
        center = tensor2ndarray(center)
        check_type('radius', radius, (np.ndarray, torch.Tensor))
        radius = tensor2ndarray(radius)
        if len(center.shape) == 1:
            center = center[None]
        assert center.shape == (radius.shape[0], 2), (
            'The shape of `center` should be (radius.shape, 2), '
            f'but got {center.shape}')
        if not (self._is_posion_valid(center -
                                      np.tile(radius.reshape((-1, 1)), (1, 2)))
                and self._is_posion_valid(
                    center + np.tile(radius.reshape((-1, 1)), (1, 2)))):
            warnings.warn(
                'Warning: The circle is out of bounds,'
                ' the drawn circle may not be in the image', UserWarning)

        center = center.tolist()
        radius = radius.tolist()
        edge_colors = color_val_matplotlib(edge_colors)  # type: ignore
        face_colors = color_val_matplotlib(face_colors)  # type: ignore
        circles = []
        for i in range(len(center)):
            circles.append(Circle(tuple(center[i]), radius[i]))

        if isinstance(line_widths, (int, float)):
            line_widths = [line_widths] * len(circles)
        line_widths = [
            min(max(linewidth, 1), self._default_font_size / 4)
            for linewidth in line_widths
        ]
        p = PatchCollection(
            circles,
            alpha=alpha,
            facecolors=face_colors,
            edgecolors=edge_colors,
            linewidths=line_widths,
            linestyles=line_styles)
        self.ax_save.add_collection(p)
        return self

    @master_only
    def draw_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor],
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[int, float] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw with
                the format of(x1,y1,x2,y2).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'g'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of bboxes.
                Defaults to 0.8.
        """
        check_type('bboxes', bboxes, (np.ndarray, torch.Tensor))
        bboxes = tensor2ndarray(bboxes)

        if len(bboxes.shape) == 1:
            bboxes = bboxes[None]
        assert bboxes.shape[-1] == 4, (
            f'The shape of `bboxes` should be (N, 4), but got {bboxes.shape}')

        assert (bboxes[:, 0] <= bboxes[:, 2]).all() and (bboxes[:, 1] <=
                                                         bboxes[:, 3]).all()
        if not self._is_posion_valid(bboxes.reshape((-1, 2, 2))):
            warnings.warn(
                'Warning: The bbox is out of bounds,'
                ' the drawn bbox may not be in the image', UserWarning)
        poly = np.stack(
            (bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 1],
             bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 3]),
            axis=-1).reshape(-1, 4, 2)
        poly = [p for p in poly]
        return self.draw_polygons(
            poly,
            alpha=alpha,
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)

    @master_only
    def draw_polygons(
        self,
        polygons: Union[Union[np.ndarray, torch.Tensor],
                        List[Union[np.ndarray, torch.Tensor]]],
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none',
        alpha: Union[int, float] = 0.8,
    ) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            polygons (Union[Union[np.ndarray, torch.Tensor],\
                List[Union[np.ndarray, torch.Tensor]]]): The polygons to draw
                with the format of (x1,y1,x2,y2,...,xn,yn).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of polygons. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value,
                all the lines will have the same colors. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
                Defaults to 'g.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors. Defaults to None.
            alpha (Union[int, float]): The transparency of polygons.
                Defaults to 0.8.
        """
        from matplotlib.collections import PolyCollection
        check_type('polygons', polygons, (list, np.ndarray, torch.Tensor))
        edge_colors = color_val_matplotlib(edge_colors)  # type: ignore
        face_colors = color_val_matplotlib(face_colors)  # type: ignore

        if isinstance(polygons, (np.ndarray, torch.Tensor)):
            polygons = [polygons]
        if isinstance(polygons, list):
            for polygon in polygons:
                assert polygon.shape[1] == 2, (
                    'The shape of each polygon in `polygons` should be (M, 2),'
                    f' but got {polygon.shape}')
        polygons = [tensor2ndarray(polygon) for polygon in polygons]
        for polygon in polygons:
            if not self._is_posion_valid(polygon):
                warnings.warn(
                    'Warning: The polygon is out of bounds,'
                    ' the drawn polygon may not be in the image', UserWarning)
        if isinstance(line_widths, (int, float)):
            line_widths = [line_widths] * len(polygons)
        line_widths = [
            min(max(linewidth, 1), self._default_font_size / 4)
            for linewidth in line_widths
        ]
        polygon_collection = PolyCollection(
            polygons,
            alpha=alpha,
            facecolor=face_colors,
            linestyles=line_styles,
            edgecolors=edge_colors,
            linewidths=line_widths)

        self.ax_save.add_collection(polygon_collection)
        return self

    @master_only
    def draw_binary_masks(
            self,
            binary_masks: Union[np.ndarray, torch.Tensor],
            colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            alphas: Union[float, List[float]] = 0.8) -> 'Visualizer':
        """Draw single or multiple binary masks.

        Args:
            binary_masks (np.ndarray, torch.Tensor): The binary_masks to draw
                with of shape (N, H, W), where H is the image height and W is
                the image width. Each value in the array is either a 0 or 1
                value of uint8 type.
            colors (np.ndarray): The colors which binary_masks will convert to.
                ``colors`` can have the same length with binary_masks or just
                single value. If ``colors`` is single value, all the
                binary_masks will convert to the same colors. The colors format
                is RGB. Defaults to np.array([0, 255, 0]).
            alphas (Union[int, List[int]]): The transparency of masks.
                Defaults to 0.8.
        """
        check_type('binary_masks', binary_masks, (np.ndarray, torch.Tensor))
        binary_masks = tensor2ndarray(binary_masks)
        assert binary_masks.dtype == np.bool_, (
            'The dtype of binary_masks should be np.bool_, '
            f'but got {binary_masks.dtype}')
        binary_masks = binary_masks.astype('uint8') * 255
        img = self.get_image()
        if binary_masks.ndim == 2:
            binary_masks = binary_masks[None]
        assert img.shape[:2] == binary_masks.shape[
                                1:], '`binary_masks` must have ' \
                                     'the same shape with image'
        binary_mask_len = binary_masks.shape[0]

        check_type_and_length('colors', colors, (str, tuple, list),
                              binary_mask_len)
        colors = value2list(colors, (str, tuple), binary_mask_len)
        colors = [
            color_str2rgb(color) if isinstance(color, str) else color
            for color in colors
        ]
        for color in colors:
            assert len(color) == 3
            for channel in color:
                assert 0 <= channel <= 255  # type: ignore

        if isinstance(alphas, float):
            alphas = [alphas] * binary_mask_len

        for binary_mask, color, alpha in zip(binary_masks, colors, alphas):
            binary_mask_complement = cv2.bitwise_not(binary_mask)
            rgb = np.zeros_like(img)
            rgb[...] = color
            rgb = cv2.bitwise_and(rgb, rgb, mask=binary_mask)
            img_complement = cv2.bitwise_and(
                img, img, mask=binary_mask_complement)
            rgb = rgb + img_complement
            img = cv2.addWeighted(img, 1 - alpha, rgb, alpha, 0)
        self.ax_save.imshow(
            img,
            extent=(0, self.width, self.height, 0),
            interpolation='nearest')
        return self

    @staticmethod
    @master_only
    def draw_featmap(featmap: torch.Tensor,
                     overlaid_image: Optional[np.ndarray] = None,
                     channel_reduction: Optional[str] = 'squeeze_mean',
                     topk: int = 20,
                     arrangement: Tuple[int, int] = (4, 5),
                     resize_shape: Optional[tuple] = None,
                     alpha: float = 0.5) -> np.ndarray:
        """Draw featmap.

        - If `overlaid_image` is not None, the final output image will be the
          weighted sum of img and featmap.

        - If `resize_shape` is specified, `featmap` and `overlaid_image`
          are interpolated.

        - If `resize_shape` is None and `overlaid_image` is not None,
          the feature map will be interpolated to the spatial size of the image
          in the case where the spatial dimensions of `overlaid_image` and
          `featmap` are different.

        - If `channel_reduction` is "squeeze_mean" and "select_max",
          it will compress featmap to single channel image and weighted
          sum to `overlaid_image`.

        - If `channel_reduction` is None

          - If topk <= 0, featmap is assert to be one or three
            channel and treated as image and will be weighted sum
            to ``overlaid_image``.
          - If topk > 0, it will select topk channel to show by the sum of
            each channel. At the same time, you can specify the `arrangement`
            to set the window layout.

        Args:
            featmap (torch.Tensor): The featmap to draw which format is
                (C, H, W).
            overlaid_image (np.ndarray, optional): The overlaid image.
                Defaults to None.
            channel_reduction (str, optional): Reduce multiple channels to a
                single channel. The optional value is 'squeeze_mean'
                or 'select_max'. Defaults to 'squeeze_mean'.
            topk (int): If channel_reduction is not None and topk > 0,
                it will select topk channel to show by the sum of each channel.
                if topk <= 0, tensor_chw is assert to be one or three.
                Defaults to 20.
            arrangement (Tuple[int, int]): The arrangement of featmap when
                channel_reduction is None and topk > 0. Defaults to (4, 5).
            resize_shape (tuple, optional): The shape to scale the feature map.
                Defaults to None.
            alpha (Union[int, List[int]]): The transparency of featmap.
                Defaults to 0.5.

        Returns:
            np.ndarray: RGB image.
        """
        import matplotlib.pyplot as plt
        assert isinstance(featmap,
                          torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                          f' but got {type(featmap)}')
        assert featmap.ndim == 3, f'Input dimension must be 3, ' \
                                  f'but got {featmap.ndim}'
        featmap = featmap.detach().cpu()

        if overlaid_image is not None:
            if overlaid_image.ndim == 2:
                overlaid_image = cv2.cvtColor(overlaid_image,
                                              cv2.COLOR_GRAY2RGB)

            if overlaid_image.shape[:2] != featmap.shape[1:]:
                warnings.warn(
                    f'Since the spatial dimensions of '
                    f'overlaid_image: {overlaid_image.shape[:2]} and '
                    f'featmap: {featmap.shape[1:]} are not same, '
                    f'the feature map will be interpolated. '
                    f'This may cause mismatch problems !')
                if resize_shape is None:
                    featmap = F.interpolate(
                        featmap[None],
                        overlaid_image.shape[:2],
                        mode='bilinear',
                        align_corners=False)[0]

        if resize_shape is not None:
            featmap = F.interpolate(
                featmap[None],
                resize_shape,
                mode='bilinear',
                align_corners=False)[0]
            if overlaid_image is not None:
                overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

        if channel_reduction is not None:
            assert channel_reduction in [
                'squeeze_mean', 'select_max'], \
                f'Mode only support "squeeze_mean", "select_max", ' \
                f'but got {channel_reduction}'
            if channel_reduction == 'select_max':
                sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
                _, indices = torch.topk(sum_channel_featmap, 1)
                feat_map = featmap[indices]
            else:
                feat_map = torch.mean(featmap, dim=0)
            return convert_overlay_heatmap(feat_map, overlaid_image, alpha)
        elif topk <= 0:
            featmap_channel = featmap.shape[0]
            assert featmap_channel in [
                1, 3
            ], ('The input tensor channel dimension must be 1 or 3 '
                'when topk is less than 1, but the channel '
                f'dimension you input is {featmap_channel}, you can use the'
                ' channel_reduction parameter or set topk greater than '
                '0 to solve the error')
            return convert_overlay_heatmap(featmap, overlaid_image, alpha)
        else:
            row, col = arrangement
            channel, height, width = featmap.shape
            assert row * col >= topk, 'The product of row and col in ' \
                                      'the `arrangement` is less than ' \
                                      'topk, please set the ' \
                                      '`arrangement` correctly'

            # Extract the feature map of topk
            topk = min(channel, topk)
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, topk)
            topk_featmap = featmap[indices]

            fig = plt.figure(frameon=False)
            # Set the window layout
            fig.subplots_adjust(
                left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            dpi = fig.get_dpi()
            fig.set_size_inches((width * col + 1e-2) / dpi,
                                (height * row + 1e-2) / dpi)
            for i in range(topk):
                axes = fig.add_subplot(row, col, i + 1)
                axes.axis('off')
                axes.text(2, 15, f'channel: {indices[i]}', fontsize=10)
                axes.imshow(
                    convert_overlay_heatmap(topk_featmap[i], overlaid_image,
                                            alpha))
            image = img_from_canvas(fig.canvas)
            plt.close(fig)
            return image

    @master_only
    def add_config(self, config: Config, **kwargs):
        """Record the config.

        Args:
            config (Config): The Config object.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_config(config, **kwargs)

    @master_only
    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_graph(model, data_batch, **kwargs)

    @master_only
    def add_image(self, name: str, image: np.ndarray, step: int = 0) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, image, step)  # type: ignore

    @master_only
    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data.

        Args:
            name (str): The scalar identifier.
            value (float, int): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_scalar(name, value, step, **kwargs)  # type: ignore

    @master_only
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Defaults to None.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_scalars(scalar_dict, step, file_path, **kwargs)

    @master_only
    def add_datasample(self,
                       name,
                       image: np.ndarray,
                       data_sample: Optional['BaseDataElement'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        """Draw datasample."""
        pass

    def close(self) -> None:
        """close an opened object."""
        for vis_backend in self._vis_backends.values():
            vis_backend.close()

    @classmethod
    def get_instance(cls, name: str, **kwargs) -> 'Visualizer':
        instance = super().get_instance(name, **kwargs)
        Visualizer._instance_dict[name] = instance
        return instance
import re
@LOG_PROCESSORS.register_module()
class LogProcessor:
    def __init__(self,
                 window_size=10,
                 by_epoch=True,
                 custom_cfg: Optional[List[dict]] = None,
                 num_digits: int = 4,
                 log_with_hierarchy: bool = False,
                 mean_pattern=r'.*(loss|time|data_time|grad_norm).*'):
        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else []
        self.num_digits = num_digits
        self.log_with_hierarchy = log_with_hierarchy
        self.mean_pattern = re.compile(mean_pattern)
        self._check_custom_cfg()

    def get_log_after_iter(self, runner, batch_idx: int,
                           mode: str) -> Tuple[dict, str]:
        """Format log string after training, validation or testing iteration.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner, train, test or val.

        Return:
            Tuple[dict, str]: Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in ['train', 'test', 'val']
        # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
        parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                              self.custom_cfg)
        # log_tag is used to write log information to terminal
        log_tag = self._collect_scalars(parsed_cfg, runner, mode)

        # If `self.log_with_hierarchy` is False, the tag is the same as
        # log_tag. Otherwise, each key in tag starts with prefix `train`,
        # `test` or `val`
        if not self.log_with_hierarchy:
            tag = copy.deepcopy(log_tag)
        else:
            tag = self._collect_scalars(parsed_cfg, runner, mode, True)

        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.endswith('lr'):
                key = self._remove_prefix(key, f'{mode}/')
                log_tag.pop(key)
                lr_str_list.append(f'{key}: '
                                   f'{value:.{self.num_digits}e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
        if self.by_epoch:
            # Align the iteration log:
            # Epoch(train)  [  9][010/270]
            # ...                 ||| |||
            # Epoch(train)  [ 10][100/270]
            dataloader_len = self._get_dataloader_size(runner, mode)
            cur_iter = self._get_iter(runner, batch_idx)
            cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))
            if mode in ['train', 'val']:
                cur_epoch = self._get_epoch(runner, mode)
                if not (isinstance(runner._train_loop, dict)
                        or runner._train_loop is None):
                    # Right Align the epoch log:
                    # Epoch(train)   [9][100/270]
                    # ...             ||
                    # Epoch(train) [100][100/270]
                    max_epochs = runner.max_epochs
                    # 3 means the three characters: "[", "]", and " " occupied
                    # in " [{max_epochs}]"
                    cur_epoch_str = f'[{cur_epoch}]'.rjust(
                        len(str(max_epochs)) + 3, ' ')
                else:
                    cur_epoch_str = f'[{cur_epoch}]'
                tag['epoch'] = cur_epoch
                log_str = (f'Epoch({mode}){cur_epoch_str}'
                           f'[{cur_iter_str}/{dataloader_len}]  ')
            else:
                log_str = (f'Epoch({mode}) '
                           f'[{cur_iter_str}/{dataloader_len}]  ')
        else:
            if mode == 'train':
                cur_iter = self._get_iter(runner, batch_idx)
                cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
                log_str = (f'Iter({mode}) '
                           f'[{cur_iter_str}/{runner.max_iters}]  ')
            else:
                dataloader_len = self._get_dataloader_size(runner, mode)
                cur_iter_str = str(batch_idx + 1).rjust(
                    len(str(dataloader_len)))
                log_str = (f'Iter({mode}) [{cur_iter_str}/{dataloader_len}]  ')
        # Add global iter.
        if isinstance(runner._train_loop, dict) or runner._train_loop is None:
            tag['iter'] = 0
        else:
            tag['iter'] = runner.iter + 1
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str}  '
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if (all(item in log_tag for item in ['time', 'data_time'])
                and 'eta' in runner.message_hub.runtime_info):
            eta = runner.message_hub.get_info('eta')
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += f'eta: {eta_str}  '
            log_str += (f'time: {log_tag["time"]:.{self.num_digits}f}  '
                        f'data_time: '
                        f'{log_tag["data_time"]:.{self.num_digits}f}  ')
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda/musa is available,
        # the max memory occupied should be calculated.
        if is_cuda_available() or is_musa_available():
            max_memory = self._get_max_memory(runner)
            log_str += f'memory: {max_memory}  '
            tag['memory'] = max_memory

        # Loop left keys to fill `log_str`.
        if mode in ('train', 'val'):
            log_items = []
            for name, val in log_tag.items():
                if mode == 'val' and not name.startswith('val/loss'):
                    continue
                if isinstance(val, float):
                    val = f'{val:.{self.num_digits}f}'
                log_items.append(f'{name}: {val}')
            log_str += '  '.join(log_items)
        return tag, log_str

    def get_log_after_epoch(self,
                            runner,
                            batch_idx: int,
                            mode: str,
                            with_non_scalar: bool = False) -> Tuple[dict, str]:
        """Format log string after validation or testing epoch.

        Args:
            runner (Runner): The runner of validation/testing phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner.
            with_non_scalar (bool): Whether to include non-scalar infos in the
                returned tag. Defaults to False.

        Return:
            Tuple[dict, str]: Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in [
            'test', 'val'
        ], ('`_get_metric_log_str` only accept val or test mode, but got '
            f'{mode}')
        dataloader_len = self._get_dataloader_size(runner, mode)

        # By epoch:
        #     Epoch(val) [10][1000/1000]  ...
        #     Epoch(test) [1000/1000] ...
        # By iteration:
        #     Iteration(val) [1000/1000]  ...
        #     Iteration(test) [1000/1000]  ...
        if self.by_epoch:
            if mode == 'val':
                cur_epoch = self._get_epoch(runner, mode)
                log_str = (f'Epoch({mode}) [{cur_epoch}][{dataloader_len}/'
                           f'{dataloader_len}]  ')
            else:
                log_str = (
                    f'Epoch({mode}) [{dataloader_len}/{dataloader_len}]  ')

        else:
            log_str = (f'Iter({mode}) [{dataloader_len}/{dataloader_len}]  ')

        custom_cfg_copy = copy.deepcopy(self.custom_cfg)
        # remove prefix
        custom_keys = [
            self._remove_prefix(cfg['data_src'], f'{mode}/')
            for cfg in custom_cfg_copy
        ]
        # Count the averaged time and data_time by epoch
        if 'time' not in custom_keys:
            custom_cfg_copy.append(
                dict(data_src='time', window_size='epoch', method_name='mean'))
        if 'data_time' not in custom_keys:
            custom_cfg_copy.append(
                dict(
                    data_src='data_time',
                    window_size='epoch',
                    method_name='mean'))
        parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                              custom_cfg_copy)
        # tag is used to write log information to different backends.
        ori_tag = self._collect_scalars(parsed_cfg, runner, mode,
                                        self.log_with_hierarchy)
        non_scalar_tag = self._collect_non_scalars(runner, mode)
        # move `time` or `data_time` to the end of the log
        tag = OrderedDict()
        time_tag = OrderedDict()
        for key, value in ori_tag.items():
            if key in (f'{mode}/time', f'{mode}/data_time', 'time',
                       'data_time'):
                time_tag[key] = value
            else:
                tag[key] = value
        # Log other messages.
        log_items = []
        log_str += '  '
        for name, val in chain(tag.items(), non_scalar_tag.items(),
                               time_tag.items()):
            if isinstance(val, float):
                val = f'{val:.{self.num_digits}f}'
            if isinstance(val, (torch.Tensor, np.ndarray)):
                # newline to display tensor and array.
                val = f'\n{val}\n'
            log_items.append(f'{name}: {val}')
        log_str += '  '.join(log_items)

        if with_non_scalar:
            tag.update(non_scalar_tag)
        tag.update(time_tag)
        return tag, log_str

    def _collect_scalars(self,
                         custom_cfg: List[dict],
                         runner,
                         mode: str,
                         reserve_prefix: bool = False) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            custom_cfg (List[dict]): A copy of ``self.custom_cfg`` with int
                ``window_size``.
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.
            reserve_prefix (bool): Whether to reserve the prefix of the key.

        Returns:
            dict: Statistical values of logs.
        """
        custom_cfg = copy.deepcopy(custom_cfg)
        tag = OrderedDict()
        # history_scalars of train/val/test phase.
        history_scalars = runner.message_hub.log_scalars
        # corresponding mode history_scalars
        mode_history_scalars = OrderedDict()
        # extract log scalars and remove prefix to `mode_history_scalars`
        # according to mode.
        for prefix_key, log_buffer in history_scalars.items():
            if prefix_key.startswith(mode):
                if not reserve_prefix:
                    key = self._remove_prefix(prefix_key, f'{mode}/')
                else:
                    key = prefix_key
                mode_history_scalars[key] = log_buffer
        for key in mode_history_scalars:
            # Update the latest learning rate and smoothed time logs.
            if re.search(self.mean_pattern, key) is not None:
                tag[key] = mode_history_scalars[key].mean(self.window_size)
            else:
                # Default statistic method is current.
                tag[key] = mode_history_scalars[key].current()
        # Update custom keys.
        for log_cfg in custom_cfg:
            data_src = log_cfg.pop('data_src')
            log_name = log_cfg.pop('log_name', data_src)
            if reserve_prefix:
                data_src = f'{mode}/{data_src}'
                log_name = f'{mode}/{log_name}'
            # log item in custom_cfg could only exist in train or val
            # mode.
            if data_src in mode_history_scalars:
                tag[log_name] = mode_history_scalars[data_src].statistics(
                    **log_cfg)
        return tag

    def _collect_non_scalars(self, runner, mode: str) -> dict:
        """Collect log information to compose a dict according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            dict: non-scalar infos of the specified mode.
        """
        # infos of train/val/test phase.
        infos = runner.message_hub.runtime_info
        # corresponding mode infos
        mode_infos = OrderedDict()
        # extract log info and remove prefix to `mode_infos` according to mode.
        for prefix_key, value in infos.items():
            if prefix_key.startswith(mode):
                if self.log_with_hierarchy:
                    key = prefix_key
                else:
                    key = self._remove_prefix(prefix_key, f'{mode}/')
                mode_infos[key] = value
        return mode_infos

    def _remove_prefix(self, string: str, prefix: str):
        """Remove the prefix ``train``, ``val`` and ``test`` of the key."""
        if string.startswith(prefix):
            return string[len(prefix):]
        else:
            return string

    def _check_custom_cfg(self) -> None:
        """Check the legality of ``self.custom_cfg``."""

        def _check_window_size():
            for log_cfg in self.custom_cfg:
                if not self.by_epoch:
                    assert log_cfg['window_size'] != 'epoch', \
                        'window_size cannot be epoch if LoggerHook.by_epoch' \
                        ' is False.'

        def _check_repeated_log_name():
            # The `log_name` of the same data_src should not be repeated.
            # If `log_name` is not specified, `data_src` will be overwritten.
            # But only allowed to be overwritten once.
            check_set = set()
            for log_cfg in self.custom_cfg:
                assert 'data_src' in log_cfg
                data_src = log_cfg['data_src']
                log_name = log_cfg.get('log_name', data_src)
                assert log_name not in check_set, (
                    f'Found duplicate {log_name} for {data_src}. Please check'
                    'your `custom_cfg` for `log_processor`. You should '
                    f'neither define duplicate `{log_name}` for {data_src} '
                    f'nor do not define any {log_name} for multiple '
                    f'{data_src}, See more information in the docstring of '
                    'LogProcessor')

                check_set.add(log_name)

        _check_repeated_log_name()
        _check_window_size()

    def _parse_windows_size(self,
                            runner,
                            batch_idx: int,
                            custom_cfg: Optional[list] = None) -> list:
        """Parse window_size defined in custom_cfg to int value.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current dataloader.
            custom_cfg (list): A copy of ``self.custom_cfg``. Defaults to None
                to keep backward compatibility.
        """
        if custom_cfg is None:
            custom_cfg = copy.deepcopy(self.custom_cfg)
        else:
            custom_cfg = copy.deepcopy(custom_cfg)
        for log_cfg in custom_cfg:
            window_size = log_cfg.get('window_size', None)
            if window_size is None or isinstance(window_size, int):
                continue
            elif window_size == 'epoch':
                log_cfg['window_size'] = batch_idx + 1
            elif window_size == 'global':
                log_cfg['window_size'] = runner.iter + 1
            else:
                raise TypeError(
                    'window_size should be int, epoch or global, but got '
                    f'invalid {window_size}')
        return custom_cfg

    def _get_max_memory(self, runner) -> int:
        """Returns the maximum GPU memory occupied by tensors in megabytes (MB)
        for a given device.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.

        Returns:
            The maximum GPU memory occupied by tensors in megabytes for a given
            device.
        """

        device = getattr(runner.model, 'output_device', None)

        if is_musa_available():
            return get_max_musa_memory(device)
        return get_max_cuda_memory(device)

    def _get_iter(self, runner, batch_idx: int) -> int:
        """Get current iteration index.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            batch_idx (int): The iteration index of current
                dataloader. Defaults to None.

        Returns:
            int: The current global iter or inner iter.
        """
        if self.by_epoch:
            current_iter = batch_idx + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def _get_epoch(self, runner, mode: str) -> int:
        """Get current epoch according to mode.

        Args:
            runner (Runner): The runner of the training/testing/validation
                process.
            mode (str): Current mode of runner.

        Returns:
            int: The current epoch.
        """
        if mode == 'train':
            epoch = runner.epoch + 1
        elif mode == 'val':
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                epoch = 0
            else:
                # normal val mode
                # runner.epoch += 1 has been done before validation
                epoch = runner.epoch
        else:
            raise ValueError(
                f"runner mode should be 'train' or 'val', but got {mode}")
        return epoch

    def _get_cur_loop(self, runner, mode: str):
        """Get current loop according to mode.

        Args:
            runner (Runner): The runner of the training/validation/testing
                process.
            mode (str): Current mode of runner.

        Returns:
            BaseLoop: Current loop of runner.
        """
        # returns type hint will occur circular import
        if mode == 'train':
            return runner.train_loop
        elif mode == 'val':
            return runner.val_loop
        else:
            return runner.test_loop

    def _get_dataloader_size(self, runner, mode) -> int:
        """Get dataloader size of current loop.

        Args:
            runner (Runner): The runner of the training/validation/testing
            mode (str): Current mode of runner.

        Returns:
            int: The dataloader size of current loop.
        """
        return len(self._get_cur_loop(runner=runner, mode=mode).dataloader)
    
from enum import Enum
from typing import Union

class Priority(Enum):

    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 30
    ABOVE_NORMAL = 40
    NORMAL = 50
    BELOW_NORMAL = 60
    LOW = 70
    VERY_LOW = 90
    LOWEST = 100


def get_priority(priority: Union[int, str, Priority]) -> int:
    """Get priority value.

    Args:
        priority (int or str or :obj:`Priority`): Priority.

    Returns:
        int: The priority value.
    """
    if isinstance(priority, int):
        if priority < 0 or priority > 100:
            raise ValueError('priority must be between 0 and 100')
        return priority
    elif isinstance(priority, Priority):
        return priority.value
    elif isinstance(priority, str):
        return Priority[priority.upper()].value
    else:
        raise TypeError('priority must be an integer or Priority enum value')
    
import torch.multiprocessing as mp

def set_multi_processing(mp_start_method: str = 'fork',
                         opencv_num_threads: int = 0,
                         distributed: bool = False) -> None:
    """Set multi-processing related environment.

    Args:
        mp_start_method (str): Set the method which should be used to start
            child processes. Defaults to 'fork'.
        opencv_num_threads (int): Number of threads for opencv.
            Defaults to 0.
        distributed (bool): True if distributed environment.
            Defaults to False.
    """
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        current_method = mp.get_start_method(allow_none=True)
        if (current_method is not None and current_method != mp_start_method):
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can '
                'change this behavior by changing `mp_start_method` in '
                'your config.')
        mp.set_start_method(mp_start_method, force=True)

    try:
        import cv2

        # disable opencv multithreading to avoid system being overloaded
        cv2.setNumThreads(opencv_num_threads)
    except ImportError:
        pass

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if 'OMP_NUM_THREADS' not in os.environ and distributed:
        omp_num_threads = 1
        warnings.warn(
            'Setting OMP_NUM_THREADS environment variable for each process'
            f' to be {omp_num_threads} in default, to avoid your system '
            'being overloaded, please further tune the variable for '
            'optimal performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and distributed:
        mkl_num_threads = 1
        warnings.warn(
            'Setting MKL_NUM_THREADS environment variable for each process'
            f' to be {mkl_num_threads} in default, to avoid your system '
            'being overloaded, please further tune the variable for '
            'optimal performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

from custom.registry_base import Registry
def is_model_wrapper(model: nn.Module, registry: Registry = MODEL_WRAPPERS):
    """Check if a module is a model wrapper.

    The following 4 model in MMEngine (and their subclasses) are regarded as
    model wrappers: DataParallel, DistributedDataParallel,
    MMDataParallel, MMDistributedDataParallel. You may add you own
    model wrapper by registering it to ``mmengine.registry.MODEL_WRAPPERS``.

    Args:
        model (nn.Module): The model to be checked.
        registry (Registry): The parent registry to search for model wrappers.

    Returns:
        bool: True if the input model is a model wrapper.
    """
    module_wrappers = tuple(registry.module_dict.values())
    if isinstance(model, module_wrappers):
        return True

    if not registry.children:
        return False

    return any(
        is_model_wrapper(model, child) for child in registry.children.values())

def default_worker_init_fn(worker_id: int,
                   num_workers: int,
                   rank: int,
                   seed: int,
                   disable_subprocess_warning: bool = False) -> None:
    """This function will be called on each worker subprocess after seeding and
    before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1].
        num_workers (int): How many subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in
            non-distributed environment, it is a constant number `0`.
        seed (int): Random seed.
    """
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if disable_subprocess_warning and worker_id != 0:
        warnings.simplefilter('ignore')

@RUNNERS.register_module()
class Runner:
    cfg: Config
    _train_loop: Optional[Union[BaseLoop, Dict]]
    _val_loop: Optional[Union[BaseLoop, Dict]]
    _test_loop: Optional[Union[BaseLoop, Dict]]

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        work_dir: str,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        auto_scale_lr: Optional[Dict] = None,
        optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        data_preprocessor: Union[nn.Module, Dict, None] = None,
        load_from: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
        default_scope: str = 'mmengine',
        randomness: Dict = dict(seed=None),
        experiment_name: Optional[str] = None,
        cfg: Optional[ConfigType] = None,
    ):
        self._work_dir = osp.abspath(work_dir)
        # mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        # breakpoint()
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
        # `dict` with single optimizer, parsed param_scheduler will be a
        # list of parameter schedulers. If `optim_wrapper` is
        # a `dict` with multiple optimizers, parsed `param_scheduler` will be
        # dict with multiple list of parameter schedulers.
        self._check_scheduler_cfg(param_scheduler)
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)
        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        # mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        if default_scope is not None:
            default_scope = DefaultScope.get_instance(  # type: ignore
                self._experiment_name,
                scope_name=default_scope)
        self.default_scope = default_scope

        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        # self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        # self.logger = self.build_logger(log_level=log_level)

        # Collect and log environment information.
        # self._log_env(env_cfg)

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        # self.visualizer = self.build_visualizer(visualizer)
        # if self.cfg:
        #     self.visualizer.add_config(self.cfg)

        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        # build a model
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)
        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)
        # print(self.model)
        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        # self.register_hooks(default_hooks, custom_hooks)
        # log hooks information
        # self.logger.info(f'Hooks will be executed in the following '
        #                  f'order:\n{self.get_hooks_info()}')

        # dump `cfg` to `work_dir`
        # self.dump_config()

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner

    @property
    def experiment_name(self):
        """str: Name of experiment."""
        return self._experiment_name

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def work_dir(self):
        """str: The working directory to save checkpoints and logs."""
        return self._work_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.max_epochs
        else:
            return 0

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.max_iters
        else:
            return 0

    @property
    def epoch(self):
        """int: Current epoch."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.epoch
        else:
            return 0

    @property
    def iter(self):
        """int: Current iteration."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.iter
        else:
            return 0

    @property
    def launcher(self):
        """str: Way to launcher multi processes."""
        return self._launcher

    @property
    def distributed(self):
        """bool: Whether current environment is distributed."""
        return self._distributed

    @property
    def rank(self):
        """int: Rank of current process."""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job."""
        return self._world_size

    @property
    def deterministic(self):
        """int: Whether cudnn to select deterministic algorithms."""
        return self._deterministic

    @property
    def seed(self):
        """int: A number to set random modules."""
        return self._seed

    @property
    def timestamp(self):
        """str: Timestamp when creating experiment."""
        return self._timestamp

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def train_loop(self):
        """:obj:`BaseLoop`: A loop to run training."""
        if isinstance(self._train_loop, BaseLoop) or self._train_loop is None:
            return self._train_loop
        else:
            self._train_loop = self.build_train_loop(self._train_loop)
            return self._train_loop

    @property
    def val_loop(self):
        """:obj:`BaseLoop`: A loop to run validation."""
        if isinstance(self._val_loop, BaseLoop) or self._val_loop is None:
            return self._val_loop
        else:
            self._val_loop = self.build_val_loop(self._val_loop)
            return self._val_loop

    @property
    def test_loop(self):
        """:obj:`BaseLoop`: A loop to run testing."""
        if isinstance(self._test_loop, BaseLoop) or self._test_loop is None:
            return self._test_loop
        else:
            self._test_loop = self.build_test_loop(self._test_loop)
            return self._test_loop

    @property
    def train_dataloader(self):
        """The data loader for training."""
        return self.train_loop.dataloader

    @property
    def val_dataloader(self):
        """The data loader for validation."""
        return self.val_loop.dataloader

    @property
    def test_dataloader(self):
        """The data loader for testing."""
        return self.test_loop.dataloader

    @property
    def val_evaluator(self):
        """:obj:`Evaluator`: An evaluator for validation."""
        return self.val_loop.evaluator

    @property
    def test_evaluator(self):
        """:obj:`Evaluator`: An evaluator for testing."""
        return self.test_loop.evaluator

    @property
    def val_interval(self):
        """int: Interval to run validation during training."""
        return self.train_loop.val_interval

    @property
    def val_begin(self):
        """int: The epoch/iteration to start running validation during
        training."""
        return self.train_loop.val_begin

    def setup_env(self, env_cfg: Dict) -> None:
        """Setup environment.

        An example of ``env_cfg``::

            env_cfg = dict(
                cudnn_benchmark=True,
                mp_cfg=dict(
                    mp_start_method='fork',
                    opencv_num_threads=0
                ),
                dist_cfg=dict(backend='nccl', timeout=1800),
                resource_limit=4096
            )

        Args:
            env_cfg (dict): Config for setting environment.
        """
        if env_cfg.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        mp_cfg: dict = env_cfg.get('mp_cfg', {})
        set_multi_processing(**mp_cfg, distributed=self.distributed)

        # init distributed env first, since logger depends on the dist info.
        if self.distributed and not is_distributed():
            dist_cfg: dict = env_cfg.get('dist_cfg', {})
            init_dist(self.launcher, **dist_cfg)

        self._rank, self._world_size = get_dist_info()

        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        # broadcast timestamp from 0 process to other processes
        # broadcast(timestamp)
        self._timestamp = time.strftime('%Y%m%d_%H%M%S',
                                        time.localtime(timestamp.item()))

        # https://github.com/pytorch/pytorch/issues/973
        # set resource limit
        if platform.system() != 'Windows':
            import resource
            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            base_soft_limit = rlimit[0]
            hard_limit = rlimit[1]
            soft_limit = min(
                max(env_cfg.get('resource_limit', 4096), base_soft_limit),
                hard_limit)
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (soft_limit, hard_limit))

    def set_randomness(self,
                       seed,
                       diff_rank_seed: bool = False,
                       deterministic: bool = False) -> None:
        """Set random seed to guarantee reproducible results.

        Args:
            seed (int): A number to set random modules.
            diff_rank_seed (bool): Whether or not set different seeds according
                to global rank. Defaults to False.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Defaults to False.
                See https://pytorch.org/docs/stable/notes/randomness.html for
                more details.
        """
        self._deterministic = deterministic
        self._seed = set_random_seed(
            seed=seed,
            deterministic=deterministic,
            diff_rank_seed=diff_rank_seed)
    
    # def build_logger(self,
    #                  log_level: Union[int, str] = 'INFO',
    #                  log_file: str = None,
    #                  **kwargs) -> MMLogger:
    #     """Build a global asscessable MMLogger.

    #     Args:
    #         log_level (int or str): The log level of MMLogger handlers.
    #             Defaults to 'INFO'.
    #         log_file (str, optional): Path of filename to save log.
    #             Defaults to None.
    #         **kwargs: Remaining parameters passed to ``MMLogger``.

    #     Returns:
    #         MMLogger: A MMLogger object build from ``logger``.
    #     """
    #     if log_file is None:
    #         log_file = osp.join(self._log_dir, f'{self.timestamp}.log')

    #     log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
    #     log_cfg.setdefault('name', self._experiment_name)
    #     # `torch.compile` in PyTorch 2.0 could close all user defined handlers
    #     # unexpectedly. Using file mode 'a' can help prevent abnormal
    #     # termination of the FileHandler and ensure that the log file could
    #     # be continuously updated during the lifespan of the runner.
    #     log_cfg.setdefault('file_mode', 'a')

    #     return MMLogger.get_instance(**log_cfg)  # type: ignore
    
    def build_message_hub(self,
                          message_hub: Optional[Dict] = None) -> MessageHub:
        """Build a global asscessable MessageHub.

        Args:
            message_hub (dict, optional): A dict to build MessageHub object.
                If not specified, default config will be used to build
                MessageHub object. Defaults to None.

        Returns:
            MessageHub: A MessageHub object build from ``message_hub``.
        """
        if message_hub is None:
            message_hub = dict(name=self._experiment_name)
        elif isinstance(message_hub, dict):
            # ensure message_hub containing name key
            message_hub.setdefault('name', self._experiment_name)
        else:
            raise TypeError(
                f'message_hub should be dict or None, but got {message_hub}')

        return MessageHub.get_instance(**message_hub)

    def build_visualizer(
            self,
            visualizer: Optional[Union[Visualizer,
                                       Dict]] = None) -> Visualizer:
        """Build a global asscessable Visualizer.

        Args:
            visualizer (Visualizer or dict, optional): A Visualizer object
                or a dict to build Visualizer object. If ``visualizer`` is a
                Visualizer object, just returns itself. If not specified,
                default config will be used to build Visualizer object.
                Defaults to None.

        Returns:
            Visualizer: A Visualizer object build from ``visualizer``.
        """
        if visualizer is None:
            visualizer = dict(
                name=self._experiment_name,
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=self._log_dir)
            return Visualizer.get_instance(**visualizer)

        if isinstance(visualizer, Visualizer):
            return visualizer

        if isinstance(visualizer, dict):
            # ensure visualizer containing name key
            visualizer.setdefault('name', self._experiment_name)
            visualizer.setdefault('save_dir', self._log_dir)
            return VISUALIZERS.build(visualizer)
        else:
            raise TypeError(
                'visualizer should be Visualizer object, a dict or None, '
                f'but got {visualizer}')

    def build_model(self, model: Union[nn.Module, Dict]) -> nn.Module:
        """Build model.

        If ``model`` is a dict, it will be used to build a nn.Module object.
        Else, if ``model`` is a nn.Module object it will be returned directly.

        An example of ``model``::

            model = dict(type='ResNet')

        Args:
            model (nn.Module or dict): A ``nn.Module`` object or a dict to
                build nn.Module object. If ``model`` is a nn.Module object,
                just returns itself.

        Note:
            The returned model must implement ``train_step``, ``test_step``
            if ``runner.train`` or ``runner.test`` will be called. If
            ``runner.val`` will be called or ``val_cfg`` is configured,
            model must implement `val_step`.

        Returns:
            nn.Module: Model build from ``model``.
        """
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, dict):
            model = MODELS.build(model)
            return model  # type: ignore
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')

    def wrap_model(
            self, model_wrapper_cfg: Optional[Dict],
            model: nn.Module) -> Union[DistributedDataParallel, nn.Module]:
        """Wrap the model to :obj:`MMDistributedDataParallel` or other custom
        distributed data-parallel module wrappers.

        An example of ``model_wrapper_cfg``::

            model_wrapper_cfg = dict(
                broadcast_buffers=False,
                find_unused_parameters=False
            )

        Args:
            model_wrapper_cfg (dict, optional): Config to wrap model. If not
                specified, ``DistributedDataParallel`` will be used in
                distributed environment. Defaults to None.
            model (nn.Module): Model to be wrapped.

        Returns:
            nn.Module or DistributedDataParallel: nn.Module or subclass of
            ``DistributedDataParallel``.
        """
        if is_model_wrapper(model):
            if model_wrapper_cfg is not None:
                raise TypeError(
                    'model has been wrapped and "model_wrapper_cfg" should be '
                    f'None, but got {model_wrapper_cfg}')

            return model

        # Set `export CUDA_VISIBLE_DEVICES=-1` to enable CPU training.
        model = model.to(get_device())

        if not self.distributed:
            # self.logger.info(
            #     'Distributed training is not used, all SyncBatchNorm (SyncBN) '
            #     'layers in the model will be automatically reverted to '
            #     'BatchNormXd layers if they are used.')
            model = revert_sync_batchnorm(model)
            return model  # type: ignore
        else:
            sync_bn = self.cfg.get('sync_bn', None)
            if sync_bn is not None:
                try:
                    model = convert_sync_batchnorm(model, sync_bn)
                except ValueError as e:
                    # self.logger.error('cfg.sync_bn should be "torch" or '
                    #                   f'"mmcv", but got {sync_bn}')
                    raise e
        if model_wrapper_cfg is None:
            find_unused_parameters = self.cfg.get('find_unused_parameters',
                                                  False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            # TODO: may use a more elegant way to get local device ID.
            model = MMDistributedDataParallel(
                module=model,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model_wrapper_cfg.setdefault('type', 'MMDistributedDataParallel')
            model_wrapper_type = MODEL_WRAPPERS.get(
                model_wrapper_cfg.get('type'))  # type: ignore
            default_args: dict = dict()
            if issubclass(
                    model_wrapper_type,  # type: ignore
                    DistributedDataParallel):
                default_args['device_ids'] = [int(os.environ['LOCAL_RANK'])]
            default_args['module'] = model
            model = MODEL_WRAPPERS.build(
                model_wrapper_cfg, default_args=default_args)
        return model

    def _init_model_weights(self) -> None:
        """Initialize the model weights if the model has
        :meth:`init_weights`"""
        model = self.model.module if is_model_wrapper(
            self.model) else self.model
        if hasattr(model, 'init_weights'):
            model.init_weights()
            # sync params and buffers
            for name, params in model.state_dict().items():
                broadcast(params)

    def scale_lr(self,
                 optim_wrapper: OptimWrapper,
                 auto_scale_lr: Optional[Dict] = None) -> None:
        """Automatically scaling learning rate in training according to the
        ratio of ``base_batch_size`` in ``autoscalelr_cfg`` and real batch
        size.

        It scales the learning rate linearly according to the
        `paper <https://arxiv.org/abs/1706.02677>`_.

        Note:
            ``scale_lr`` must be called after building optimizer wrappers
            and before building parameter schedulers.

        Args:
            optim_wrapper (OptimWrapper): An OptimWrapper object whose
                parameter groups' learning rate need to be scaled.
            auto_scale_lr (Dict, Optional): Config to scale the learning
                rate automatically. It includes ``base_batch_size`` and
                ``enable``. ``base_batch_size`` is the batch size that the
                optimizer lr is based on. ``enable`` is the switch to turn on
                and off the feature.
        """
        if (auto_scale_lr is None or not auto_scale_lr.get('enable', False)):
            return None

        assert 'base_batch_size' in auto_scale_lr, \
            'Lack of `base_batch_size` in `auto_scale_lr`.'
        dataloader: Union[DataLoader, Dict] = self._train_dataloader
        bs = dataloader.batch_size if isinstance(
            dataloader, DataLoader) else dataloader['batch_size']
        real_bs = self.world_size * bs
        base_bs = auto_scale_lr['base_batch_size']
        ratio = float(real_bs) / float(base_bs)
        self.logger.info(f'LR is set based on batch size of {base_bs} '
                         f'and the current batch size is {real_bs}. '
                         f'Scaling the original LR by {ratio}.')

        def _is_built(schedulers):
            if isinstance(schedulers, dict):
                return False if 'type' in schedulers else any(
                    _is_built(s) for s in schedulers.values())
            if isinstance(schedulers, list):
                return any(_is_built(s) for s in schedulers)
            return isinstance(schedulers, _ParamScheduler)

        if _is_built(self.param_schedulers):
            raise RuntimeError('`scale_lr` should be called before building '
                               'ParamScheduler because ParamScheduler will '
                               'store initial lr from optimizer wrappers')

        assert isinstance(optim_wrapper, OptimWrapper), \
            '`scale_lr should be called after building OptimWrapper'
        wrappers = list(optim_wrapper.values()) if isinstance(
            optim_wrapper, OptimWrapperDict) else [optim_wrapper]
        for wrapper in wrappers:
            for group in wrapper.optimizer.param_groups:
                group['lr'] = group['lr'] * ratio

    def build_optim_wrapper(
        self, optim_wrapper: Union[Optimizer, OptimWrapper, Dict]
    ) -> Union[OptimWrapper, OptimWrapperDict]:
        """Build optimizer wrapper.

        If ``optim_wrapper`` is a config dict for only one optimizer,
        the keys must contain ``optimizer``, and ``type`` is optional.
        It will build a :obj:`OptimWrapper` by default.

        If ``optim_wrapper`` is a config dict for multiple optimizers, i.e.,
        it has multiple keys and each key is for an optimizer wrapper. The
        constructor must be specified since
        :obj:`DefaultOptimizerConstructor` cannot handle the building of
        training with multiple optimizers.

        If ``optim_wrapper`` is a dict of pre-built optimizer wrappers, i.e.,
        each value of ``optim_wrapper`` represents an ``OptimWrapper``
        instance. ``build_optim_wrapper`` will directly build the
        :obj:`OptimWrapperDict` instance from ``optim_wrapper``.

        Args:
            optim_wrapper (OptimWrapper or dict): An OptimWrapper object or a
                dict to build OptimWrapper objects. If ``optim_wrapper`` is an
                OptimWrapper, just return an ``OptimizeWrapper`` instance.

        Note:
            For single optimizer training, if `optim_wrapper` is a config
            dict, `type` is optional(defaults to :obj:`OptimWrapper`) and it
            must contain `optimizer` to build the corresponding optimizer.

        Examples:
            >>> # build an optimizer
            >>> optim_wrapper_cfg = dict(type='OptimWrapper', optimizer=dict(
            ...     type='SGD', lr=0.01))
            >>> # optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
            >>> # is also valid.
            >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.01
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            >>> # build optimizer without `type`
            >>> optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
            >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.01
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            >>> # build multiple optimizers
            >>> optim_wrapper_cfg = dict(
            ...    generator=dict(type='OptimWrapper', optimizer=dict(
            ...        type='SGD', lr=0.01)),
            ...    discriminator=dict(type='OptimWrapper', optimizer=dict(
            ...        type='Adam', lr=0.001))
            ...    # need to customize a multiple optimizer constructor
            ...    constructor='CustomMultiOptimizerConstructor',
            ...)
            >>> optim_wrapper = runner.optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            name: generator
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.1
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            name: discriminator
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            'discriminator': Adam (
            Parameter Group 0
                dampening: 0
                lr: 0.02
                momentum: 0
                nesterov: False
                weight_decay: 0
            )

        Important:
            If you need to build multiple optimizers, you should implement a
            MultiOptimWrapperConstructor which gets parameters passed to
            corresponding optimizers and compose the ``OptimWrapperDict``.
            More details about how to customize OptimizerConstructor can be
            found at `optimizer-docs`_.

        Returns:
            OptimWrapper: Optimizer wrapper build from ``optimizer_cfg``.
        """  # noqa: E501
        if isinstance(optim_wrapper, OptimWrapper):
            return optim_wrapper
        if isinstance(optim_wrapper, (dict, ConfigDict, Config)):
            # optimizer must be defined for single optimizer training.
            optimizer = optim_wrapper.get('optimizer', None)

            # If optimizer is a built `Optimizer` instance, the optimizer
            # wrapper should be built by `OPTIM_WRAPPERS` registry.
            if isinstance(optimizer, Optimizer):
                optim_wrapper.setdefault('type', 'OptimWrapper')
                return OPTIM_WRAPPERS.build(optim_wrapper)  # type: ignore

            # If `optimizer` is not None or `constructor` is defined, it means,
            # optimizer wrapper will be built by optimizer wrapper
            # constructor. Therefore, `build_optim_wrapper` should be called.
            if optimizer is not None or 'constructor' in optim_wrapper:
                return build_optim_wrapper(self.model, optim_wrapper)
            else:
                # if `optimizer` is not defined, it should be the case of
                # training with multiple optimizers. If `constructor` is not
                # defined either, each value of `optim_wrapper` must be an
                # `OptimWrapper` instance since `DefaultOptimizerConstructor`
                # will not handle the case of training with multiple
                # optimizers. `build_optim_wrapper` will directly build the
                # `OptimWrapperDict` instance from `optim_wrapper.`
                optim_wrappers = OrderedDict()
                for name, optim in optim_wrapper.items():
                    if not isinstance(optim, OptimWrapper):
                        raise ValueError(
                            'each item mush be an optimizer object when '
                            '"type" and "constructor" are not in '
                            f'optimizer, but got {name}={optim}')
                    optim_wrappers[name] = optim
                return OptimWrapperDict(**optim_wrappers)
        else:
            raise TypeError('optimizer wrapper should be an OptimWrapper '
                            f'object or dict, but got {optim_wrapper}')

    def _build_param_scheduler(
            self, scheduler: Union[_ParamScheduler, Dict, List],
            optim_wrapper: OptimWrapper) -> List[_ParamScheduler]:
        """Build parameter schedulers for a single optimizer.

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.
            optim_wrapper (OptimWrapper): An optimizer wrapper object is
                passed to construct ParamScheduler object.

        Returns:
            list[_ParamScheduler]: List of parameter schedulers build from
            ``scheduler``.

        Note:
            If the train loop is built, when building parameter schedulers,
            it supports setting the max epochs/iters as the default ``end``
            of schedulers, and supports converting epoch-based schedulers
            to iter-based according to the ``convert_to_iter_based`` key.
        """
        if not isinstance(scheduler, Sequence):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        param_schedulers = []
        for scheduler in schedulers:
            if isinstance(scheduler, _ParamScheduler):
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)

                # Set default end
                if isinstance(self._train_loop, BaseLoop):
                    default_end = self.max_epochs if _scheduler.get(
                        'by_epoch', True) else self.max_iters
                    _scheduler.setdefault('end', default_end)
                    self.logger.debug(
                        f'The `end` of {_scheduler["type"]} is not set. '
                        'Use the max epochs/iters of train loop as default.')

                param_schedulers.append(
                    PARAM_SCHEDULERS.build(
                        _scheduler,
                        default_args=dict(
                            optimizer=optim_wrapper,
                            epoch_length=len(self.train_dataloader))))
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')
        return param_schedulers

    def build_param_scheduler(
            self, scheduler: Union[_ParamScheduler, Dict,
                                   List]) -> ParamSchedulerType:
        """Build parameter schedulers.

        ``build_param_scheduler`` should be called after
        ``build_optim_wrapper`` because the building logic will change
        according to the number of optimizers built by the runner.
        The cases are as below:

        - Single optimizer: When only one optimizer is built and used in the
          runner, ``build_param_scheduler`` will return a list of
          parameter schedulers.
        - Multiple optimizers: When two or more optimizers are built and used
          in runner, ``build_param_scheduler`` will return a dict containing
          the same keys with multiple optimizers and each value is a list of
          parameter schedulers. Note that, if you want different optimizers to
          use different parameter schedulers to update optimizer's
          hyper-parameters, the input parameter ``scheduler`` also needs to be
          a dict and its key are consistent with multiple optimizers.
          Otherwise, the same parameter schedulers will be used to update
          optimizer's hyper-parameters.

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.

        Examples:
            >>> # build one scheduler
            >>> optim_cfg = dict(dict(type='SGD', lr=0.01))
            >>> runner.optim_wrapper = runner.build_optim_wrapper(
            >>>     optim_cfg)
            >>> scheduler_cfg = dict(type='MultiStepLR', milestones=[1, 2])
            >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
            >>> schedulers
            [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f6966290>]  # noqa: E501

            >>> # build multiple schedulers
            >>> scheduler_cfg = [
            ...    dict(type='MultiStepLR', milestones=[1, 2]),
            ...    dict(type='StepLR', step_size=1)
            ... ]
            >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
            >>> schedulers
            [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f60dd3d0>,  # noqa: E501
            <mmengine.optim.scheduler.lr_scheduler.StepLR at 0x7f70f6eb6150>]

        Above examples only provide the case of one optimizer and one scheduler
        or multiple schedulers. If you want to know how to set parameter
        scheduler when using multiple optimizers, you can find more examples
        `optimizer-docs`_.

        Returns:
            list[_ParamScheduler] or dict[str, list[_ParamScheduler]]: List of
            parameter schedulers or a dictionary contains list of parameter
            schedulers build from ``scheduler``.

        .. _optimizer-docs:
           https://mmengine.readthedocs.io/en/latest/tutorials/optim_wrapper.html
        """
        param_schedulers: ParamSchedulerType
        if not isinstance(self.optim_wrapper, OptimWrapperDict):
            # Since `OptimWrapperDict` inherits from `OptimWrapper`,
            # `isinstance(self.optim_wrapper, OptimWrapper)` cannot tell
            # whether `self.optim_wrapper` is an `OptimizerWrapper` or
            # `OptimWrapperDict` instance. Therefore, here we simply check
            # self.optim_wrapper is not an `OptimWrapperDict` instance and
            # then assert it is an OptimWrapper instance.
            assert isinstance(self.optim_wrapper, OptimWrapper), (
                '`build_optimizer` should be called before'
                '`build_param_scheduler` because the latter depends '
                'on the former')
            param_schedulers = self._build_param_scheduler(
                scheduler, self.optim_wrapper)  # type: ignore
            return param_schedulers
        else:
            param_schedulers = dict()
            for name, optimizer in self.optim_wrapper.items():
                if isinstance(scheduler, dict) and 'type' not in scheduler:
                    # scheduler is a dict and each item is a ParamScheduler
                    # object or a config to build ParamScheduler objects
                    param_schedulers[name] = self._build_param_scheduler(
                        scheduler[name], optimizer)
                else:
                    param_schedulers[name] = self._build_param_scheduler(
                        scheduler, optimizer)

            return param_schedulers

    def build_evaluator(self, evaluator: Union[Dict, List,
                                               Evaluator]) -> Evaluator:
        """Build evaluator.

        Examples of ``evaluator``::

            # evaluator could be a built Evaluator instance
            evaluator = Evaluator(metrics=[ToyMetric()])

            # evaluator can also be a list of dict
            evaluator = [
                dict(type='ToyMetric1'),
                dict(type='ToyEvaluator2')
            ]

            # evaluator can also be a list of built metric
            evaluator = [ToyMetric1(), ToyMetric2()]

            # evaluator can also be a dict with key metrics
            evaluator = dict(metrics=ToyMetric())
            # metric is a list
            evaluator = dict(metrics=[ToyMetric()])

        Args:
            evaluator (Evaluator or dict or list): An Evaluator object or a
                config dict or list of config dict used to build an Evaluator.

        Returns:
            Evaluator: Evaluator build from ``evaluator``.
        """
        if isinstance(evaluator, Evaluator):
            return evaluator
        elif isinstance(evaluator, dict):
            # if `metrics` in dict keys, it means to build customized evalutor
            if 'metrics' in evaluator:
                evaluator.setdefault('type', 'Evaluator')
                return EVALUATOR.build(evaluator)
            # otherwise, default evalutor will be built
            else:
                return Evaluator(evaluator)  # type: ignore
        elif isinstance(evaluator, list):
            # use the default `Evaluator`
            return Evaluator(evaluator)  # type: ignore
        else:
            raise TypeError(
                'evaluator should be one of dict, list of dict, and Evaluator'
                f', but got {evaluator}')

    @staticmethod
    def build_dataloader(dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        """Build dataloader.

        The method builds three components:

        - Dataset
        - Sampler
        - Dataloader

        An example of ``dataloader``::

            dataloader = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.
            seed (int, optional): Random seed. Defaults to None.
            diff_rank_seed (bool): Whether or not set different seeds to
                different ranks. If True, the seed passed to sampler is set
                to None, in order to synchronize the seeds used in samplers
                across different ranks.


        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, 'full_init'):
                dataset.full_init()
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        num_batch_per_epoch = dataloader_cfg.pop('num_batch_per_epoch', None)
        if num_batch_per_epoch is not None:
            world_size = get_world_size()
            num_samples = (
                num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
                world_size)
            dataset = _SlicedDataset(dataset, num_samples)

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg,
                default_args=dict(dataset=dataset, seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]

        if 'worker_init_fn' in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
            worker_init_fn_type = worker_init_fn_cfg.pop('type')
            if isinstance(worker_init_fn_type, str):
                worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            elif callable(worker_init_fn_type):
                worker_init_fn = worker_init_fn_type
            else:
                raise TypeError(
                    'type of worker_init_fn should be string or callable '
                    f'object, but got {type(worker_init_fn_type)}')
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn,
                              **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if ('persistent_workers' in dataloader_cfg
                and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
            print_log(
                '`persistent_workers` is only available when '
                'pytorch version >= 1.7',
                logger='current',
                level=logging.WARNING)
            dataloader_cfg.pop('persistent_workers')

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                            dict(type='pseudo_collate'))
        if isinstance(collate_fn_cfg, dict):
            collate_fn_type = collate_fn_cfg.pop('type')
            if isinstance(collate_fn_type, str):
                collate_fn = FUNCTIONS.get(collate_fn_type)
            else:
                collate_fn = collate_fn_type
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
        elif callable(collate_fn_cfg):
            collate_fn = collate_fn_cfg
        else:
            raise TypeError(
                'collate_fn should be a dict or callable object, but got '
                f'{collate_fn_cfg}')
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader

    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build training loop.

        Examples of ``loop``::

            # `EpochBasedTrainLoop` will be used
            loop = dict(by_epoch=True, max_epochs=3)

            # `IterBasedTrainLoop` will be used
            loop = dict(by_epoch=False, max_epochs=3)

            # custom training loop
            loop = dict(type='CustomTrainLoop', max_epochs=3)

        Args:
            loop (BaseLoop or dict): A training loop or a dict to build
                training loop. If ``loop`` is a training loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Training loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg and 'by_epoch' in loop_cfg:
            raise RuntimeError(
                'Only one of `type` or `by_epoch` can exist in `loop_cfg`.')

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self, dataloader=self._train_dataloader))
        else:
            by_epoch = loop_cfg.pop('by_epoch')
            if by_epoch:
                loop = EpochBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
            else:
                loop = IterBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
        return loop  # type: ignore

    def build_val_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build validation loop.

        Examples of ``loop``:

            # `ValLoop` will be used
            loop = dict()

            # custom validation loop
            loop = dict(type='CustomValLoop')

        Args:
            loop (BaseLoop or dict): A validation loop or a dict to build
                validation loop. If ``loop`` is a validation loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Validation loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'val_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self._val_dataloader,
                    evaluator=self._val_evaluator))
        else:
            loop = ValLoop(
                **loop_cfg,
                runner=self,
                dataloader=self._val_dataloader,
                evaluator=self._val_evaluator)  # type: ignore

        return loop  # type: ignore

    def build_test_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build test loop.

        Examples of ``loop``::

            # `TestLoop` will be used
            loop = dict()

            # custom test loop
            loop = dict(type='CustomTestLoop')

        Args:
            loop (BaseLoop or dict): A test loop or a dict to build test loop.
                If ``loop`` is a test loop object, just returns itself.

        Returns:
            :obj:`BaseLoop`: Test loop object build from ``loop_cfg``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'test_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)  # type: ignore

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self._test_dataloader,
                    evaluator=self._test_evaluator))
        else:
            loop = TestLoop(
                **loop_cfg,
                runner=self,
                dataloader=self._test_dataloader,
                evaluator=self._test_evaluator)  # type: ignore

        return loop  # type: ignore

    def build_log_processor(
            self, log_processor: Union[LogProcessor, Dict]) -> LogProcessor:
        """Build test log_processor.

        Examples of ``log_processor``:

            # `LogProcessor` will be used
            log_processor = dict()

            # custom log_processor
            log_processor = dict(type='CustomLogProcessor')

        Args:
            log_processor (LogProcessor or dict): A log processor or a dict
            to build log processor. If ``log_processor`` is a log processor
            object, just returns itself.

        Returns:
            :obj:`LogProcessor`: Log processor object build from
            ``log_processor_cfg``.
        """
        if isinstance(log_processor, LogProcessor):
            return log_processor
        elif not isinstance(log_processor, dict):
            raise TypeError(
                'log processor should be a LogProcessor object or dict, but'
                f'got {log_processor}')

        log_processor_cfg = copy.deepcopy(log_processor)  # type: ignore

        if 'type' in log_processor_cfg:
            log_processor = LOG_PROCESSORS.build(log_processor_cfg)
        else:
            log_processor = LogProcessor(**log_processor_cfg)  # type: ignore

        return log_processor  # type: ignore

    def get_hooks_info(self) -> str:
        # Get hooks info in each stage
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self.hooks:
            try:
                priority = Priority(hook.priority).name  # type: ignore
            except ValueError:
                priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def load_or_resume(self) -> None:
        """load or resume checkpoint."""
        if self._has_loaded:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if self._resume and self._load_from is None:
            # auto resume from the latest checkpoint
            resume_from = find_latest_checkpoint(self.work_dir)
            self.logger.info(
                f'Auto resumed from the latest checkpoint {resume_from}.')
        elif self._resume and self._load_from is not None:
            # resume from the specified checkpoint
            resume_from = self._load_from

        if resume_from is not None:
            self.resume(resume_from)
            self._has_loaded = True
        elif self._load_from is not None:
            self.load_checkpoint(self._load_from)
            self._has_loaded = True

    def train(self) -> nn.Module:
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

        if self.param_schedulers is not None:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        # initialize the model weights
        self._init_model_weights()

        # try to enable activation_checkpointing feature
        modules = self.cfg.get('activation_checkpointing', None)
        if modules is not None:
            self.logger.info(f'Enabling the "activation_checkpointing" feature'
                             f' for sub-modules: {modules}')
            turn_on_activation_checkpointing(ori_model, modules)

        # try to enable efficient_conv_bn_eval feature
        modules = self.cfg.get('efficient_conv_bn_eval', None)
        if modules is not None:
            self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                             f' for sub-modules: {modules}')
            turn_on_efficient_conv_bn_eval(ori_model, modules)

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        # Initiate inner count of `optim_wrapper`.
        self.optim_wrapper.initialize_count_status(
            self.model,
            self._train_loop.iter,  # type: ignore
            self._train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self._maybe_compile('train_step')

        model = self.train_loop.run()  # type: ignore
        self.call_hook('after_run')
        return model

    def val(self) -> dict:
        """Launch validation.

        Returns:
            dict: A dict of metrics on validation set.
        """
        if self._val_loop is None:
            raise RuntimeError(
                '`self._val_loop` should not be None when calling val method.'
                'Please provide `val_dataloader`, `val_cfg` and '
                '`val_evaluator` arguments when initializing runner.')

        self._val_loop = self.build_val_loop(self._val_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        metrics = self.val_loop.run()  # type: ignore
        self.call_hook('after_run')
        return metrics

    def test(self) -> dict:
        """Launch test.

        Returns:
            dict: A dict of metrics on testing set.
        """
        if self._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')
        return metrics

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            **kwargs: Keyword arguments passed to hook.
        """
        for hook in self._hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None

    def register_hook(
            self,
            hook: Union[Hook, Dict],
            priority: Optional[Union[str, int, Priority]] = None) -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Priority of hook will be decided with the following priority:

        - ``priority`` argument. If ``priority`` is given, it will be priority
          of hook.
        - If ``hook`` argument is a dict and ``priority`` in it, the priority
          will be the value of ``hook['priority']``.
        - If ``hook`` argument is a dict but ``priority`` not in it or ``hook``
          is an instance of ``hook``, the priority will be ``hook.priority``.

        Args:
            hook (:obj:`Hook` or dict): The hook to be registered.
            priority (int or str or :obj:`Priority`, optional): Hook priority.
                Lower value means higher priority.
        """
        if not isinstance(hook, (Hook, dict)):
            raise TypeError(
                f'hook should be an instance of Hook or dict, but got {hook}')

        _priority = None
        if isinstance(hook, dict):
            if 'priority' in hook:
                _priority = hook.pop('priority')

            hook_obj = HOOKS.build(hook)
        else:
            hook_obj = hook

        if priority is not None:
            hook_obj.priority = priority
        elif _priority is not None:
            hook_obj.priority = _priority

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if get_priority(hook_obj.priority) >= get_priority(
                    self._hooks[i].priority):
                self._hooks.insert(i + 1, hook_obj)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook_obj)

    def register_default_hooks(
            self,
            hooks: Optional[Dict[str, Union[Hook, Dict]]] = None) -> None:
        """Register default hooks into hook list.

        ``hooks`` will be registered into runner to execute some default
        actions like updating model parameters or saving checkpoints.

        Default hooks and their priorities:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | RuntimeInfoHook      | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | IterTimerHook        | NORMAL (50)             |
        +----------------------+-------------------------+
        | DistSamplerSeedHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | LoggerHook           | BELOW_NORMAL (60)       |
        +----------------------+-------------------------+
        | ParamSchedulerHook   | LOW (70)                |
        +----------------------+-------------------------+
        | CheckpointHook       | VERY_LOW (90)           |
        +----------------------+-------------------------+

        If ``hooks`` is None, above hooks will be registered by
        default::

            default_hooks = dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
            )

        If not None, ``hooks`` will be merged into ``default_hooks``.
        If there are None value in default_hooks, the corresponding item will
        be popped from ``default_hooks``::

            hooks = dict(timer=None)

        The final registered default hooks will be :obj:`RuntimeInfoHook`,
        :obj:`DistSamplerSeedHook`, :obj:`LoggerHook`,
        :obj:`ParamSchedulerHook` and :obj:`CheckpointHook`.

        Args:
            hooks (dict[str, Hook or dict], optional): Default hooks or configs
                to be registered.
        """
        default_hooks: dict = dict(
            runtime_info=dict(type='RuntimeInfoHook'),
            timer=dict(type='IterTimerHook'),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            logger=dict(type='LoggerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1),
        )
        if hooks is not None:
            for name, hook in hooks.items():
                if name in default_hooks and hook is None:
                    # remove hook from _default_hooks
                    default_hooks.pop(name)
                else:
                    assert hook is not None
                    default_hooks[name] = hook

        for hook in default_hooks.values():
            self.register_hook(hook)

    def register_custom_hooks(self, hooks: List[Union[Hook, Dict]]) -> None:
        """Register custom hooks into hook list.

        Args:
            hooks (list[Hook | dict]): List of hooks or configs to be
                registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hooks(
            self,
            default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
            custom_hooks: Optional[List[Union[Hook, Dict]]] = None) -> None:
        """Register default hooks and custom hooks into hook list.

        Args:
            default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks
                to execute default actions like updating model parameters and
                saving checkpoints.  Defaults to None.
            custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
                custom actions like visualizing images processed by pipeline.
                Defaults to None.
        """
        self.register_default_hooks(default_hooks)

        if custom_hooks is not None:
            self.register_custom_hooks(custom_hooks)

    def resume(self,
               filename: str,
               resume_optimizer: bool = True,
               resume_param_scheduler: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
            map_location (str or callable):A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'default'.
        """
        if map_location == 'default':
            device = get_device()
            checkpoint = self.load_checkpoint(filename, map_location=device)
        else:
            checkpoint = self.load_checkpoint(
                filename, map_location=map_location)

        self.train_loop._epoch = checkpoint['meta']['epoch']
        self.train_loop._iter = checkpoint['meta']['iter']

        # check whether the number of GPU used for current experiment
        # is consistent with resuming from checkpoint
        if 'config' in checkpoint['meta']:
            config = mmengine.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if (previous_gpu_ids is not None and len(previous_gpu_ids) > 0
                    and len(previous_gpu_ids) != self._world_size):
                # TODO, should we modify the iteration?
                if (self.auto_scale_lr is None
                        or not self.auto_scale_lr.get('enable', False)):
                    raise RuntimeError(
                        'Number of GPUs used for current experiment is not '
                        'consistent with the checkpoint being resumed from. '
                        'This will result in poor performance due to the '
                        'learning rate. You must set the '
                        '`auto_scale_lr` parameter for Runner and make '
                        '`auto_scale_lr["enable"]=True`.')
                else:
                    self.logger.info(
                        'Number of GPU used for current experiment is not '
                        'consistent with resuming from checkpoint but the '
                        'leaning rate will be adjusted according to the '
                        f'setting in auto_scale_lr={self.auto_scale_lr}')

        # resume random seed
        resumed_seed = checkpoint['meta'].get('seed', None)
        current_seed = self._randomness_cfg.get('seed')
        if resumed_seed is not None and resumed_seed != current_seed:
            if current_seed is not None:
                self.logger.warning(f'The value of random seed in the '
                                    f'checkpoint "{resumed_seed}" is '
                                    f'different from the value in '
                                    f'`randomness` config "{current_seed}"')
            self._randomness_cfg.update(seed=resumed_seed)
            self.set_randomness(**self._randomness_cfg)

        resumed_dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        dataset_meta = getattr(self.train_dataloader.dataset, 'metainfo', None)

        # `resumed_dataset_meta` and `dataset_meta` could be object like
        # np.ndarray, which cannot be directly judged as equal or not,
        # therefore we just compared their dumped results.
        if pickle.dumps(resumed_dataset_meta) != pickle.dumps(dataset_meta):
            self.logger.warning(
                'The dataset metainfo from the resumed checkpoint is '
                'different from the current training dataset, please '
                'check the correctness of the checkpoint or the training '
                'dataset.')

        self.message_hub.load_state_dict(checkpoint['message_hub'])

        # resume optimizer
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
            self.optim_wrapper.load_state_dict(  # type: ignore
                checkpoint['optimizer'])

        # resume param scheduler
        if resume_param_scheduler and self.param_schedulers is None:
            self.logger.warning(
                '`resume_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip resuming parameter schedulers')
            resume_param_scheduler = False
        if 'param_schedulers' in checkpoint and resume_param_scheduler:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore
            if isinstance(self.param_schedulers, dict):
                for name, schedulers in self.param_schedulers.items():
                    for scheduler, ckpt_scheduler in zip(
                            schedulers, checkpoint['param_schedulers'][name]):
                        scheduler.load_state_dict(ckpt_scheduler)
            else:
                for scheduler, ckpt_scheduler in zip(
                        self.param_schedulers,  # type: ignore
                        checkpoint['param_schedulers']):
                    scheduler.load_state_dict(ckpt_scheduler)

        self._has_loaded = True

        self.logger.info(f'resumed epoch: {self.epoch}, iter: {self.iter}')

    def load_checkpoint(self,
                        filename: str,
                        map_location: Union[str, Callable] = 'cpu',
                        strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """
        checkpoint = _load_checkpoint(filename, map_location=map_location)

        # Add comments to describe the usage of `after_load_ckpt`
        self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = _load_checkpoint_to_model(
            model, checkpoint, strict, revise_keys=revise_keys)

        self._has_loaded = True

        # self.logger.info(f'Load checkpoint from {filename}')

        return checkpoint

    @master_only
    def save_checkpoint(
        self,
        out_dir: str,
        filename: str,
        file_client_args: Optional[dict] = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: Optional[dict] = None,
        by_epoch: bool = True,
        backend_args: Optional[dict] = None,
    ):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmengine.fileio.FileClient` for
                details. Defaults to None. It will be deprecated in future.
                Please use `backend_args` instead.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Decide the number of epoch or iteration saved in
                checkpoint. Defaults to True.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.
                New in v0.2.0.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.setdefault('epoch', self.epoch + 1)
            meta.setdefault('iter', self.iter)
        else:
            meta.setdefault('epoch', self.epoch)
            meta.setdefault('iter', self.iter + 1)

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set at '
                    'the same time.')

            file_client = FileClient.infer_client(file_client_args, out_dir)
            filepath = file_client.join_path(out_dir, filename)
        else:
            filepath = join_path(  # type: ignore
                out_dir, filename, backend_args=backend_args)

        meta.update(
            cfg=self.cfg.pretty_text,
            seed=self.seed,
            experiment_name=self.experiment_name,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine_version=mmengine.__version__ + get_git_hash())

        if hasattr(self.train_dataloader.dataset, 'metainfo'):
            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = {
            'meta':
            meta,
            'state_dict':
            weights_to_cpu(model.state_dict()),
            'message_hub':
            apply_to(self.message_hub.state_dict(),
                     lambda x: hasattr(x, 'cpu'), lambda x: x.cpu()),
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optim_wrapper, OptimWrapper):
                checkpoint['optimizer'] = apply_to(
                    self.optim_wrapper.state_dict(),
                    lambda x: hasattr(x, 'cpu'), lambda x: x.cpu())
            else:
                raise TypeError(
                    'self.optim_wrapper should be an `OptimWrapper` '
                    'or `OptimWrapperDict` instance, but got '
                    f'{self.optim_wrapper}')

        # save param scheduler state dict
        if save_param_scheduler and self.param_schedulers is None:
            self.logger.warning(
                '`save_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip saving parameter schedulers')
            save_param_scheduler = False
        if save_param_scheduler:
            if isinstance(self.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in self.param_schedulers:  # type: ignore
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)

        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
        save_checkpoint(
            checkpoint,
            filepath,
            file_client_args=file_client_args,
            backend_args=backend_args)

    @master_only
    def dump_config(self) -> None:
        """Dump config to `work_dir`."""
        if self.cfg.filename is not None:
            filename = osp.basename(self.cfg.filename)
        else:
            filename = f'{self.timestamp}.py'
        self.cfg.dump(osp.join(self.work_dir, filename))

    def _check_scheduler_cfg(
            self, param_scheduler: Optional[Union[dict, list,
                                                  _ParamScheduler]]) -> None:
        """Parse `param_scheduler` to a list of parameter schedulers, or a
        `dict` of which each value is a list of parameter schedulers.

        If only one optimizer is used, the parsed config should be a
        list of parameter scheduler configs or instances. If multiple
        optimizers are used, the parsed config should be `dict`.
        Its key should be consistent with the optimizer `dict` and its value
        should be a list of parameter scheduler configs or instances. See
        :meth:`build_param_scheduler` for more details.

        Examples:
            >>> # valid scheduler:
            >>> # empty scheduler
            >>> scheduler = None
            >>> # Single scheduler
            >>> scheduler = dict(type='MultiStepLR', milestones=[1, 2])
            >>> # Single list schedulers
            >>> scheduler = [dict(type='MultiStepLR', milestones=[1, 2]),
            >>>              dict(type='MultiStepLR', milestones=[2, 3])]
            >>> # `dict` of schedulers
            >>> scheduler = dict(linear1=dict(type='MultiStepLR', milestones=[1, 2]),
            >>>                  linear2=dict(type='MultiStepLR', milestones=[1, 2]))
            >>> # `dict` of `list` of schedulers
            >>> scheduler = dict(linear1=[dict(type='MultiStepLR', milestones=[1, 2])],
            >>>                  linear2=[dict(type='MultiStepLR', milestones=[1, 2])])
            >>> # Single built scheduler
            >>> from mmengine.optim import MultiStepLR
            >>> scheduler = MultiStepLR(milestones=[1, 2], optimizer=optimizer)
            >>> # Single built list schedulers
            >>> scheduler = [MultiStepLR(milestones=[1, 2], optimizer=optimizer)]
            >>> # dict of built scheduler
            >>> scheduler = dict(linear1=MultiStepLR(milestones=[1, 2], optimizer=optimizer),
            >>>                  linear2=MultiStepLR(milestones=[1, 2], optimizer=optimizer))
            >>> # dict of built list schedulers
            >>> scheduler = dict(linear1=[MultiStepLR(milestones=[1, 2], optimizer=optimizer)],
            >>>                  linear2=[MultiStepLR(milestones=[1, 2], optimizer=optimizer)])

        Args:
            param_scheduler (dict or list): The original parameter scheduler.
        """  # noqa: E501
        if param_scheduler is None:
            return
        if isinstance(param_scheduler, _ParamScheduler):
            return
        if is_seq_of(param_scheduler, _ParamScheduler):
            return

        if is_seq_of(param_scheduler, dict):
            for _param_scheduler in param_scheduler:
                assert 'type' in _param_scheduler, (
                    'Each parameter scheduler should contain the key type, '
                    f'but got {_param_scheduler}')
        elif isinstance(param_scheduler, dict):
            if 'type' not in param_scheduler:
                for key, _param_scheduler in param_scheduler.items():
                    assert isinstance(
                        _param_scheduler,
                        (dict, tuple, list, _ParamScheduler)), (
                            'Each value of `param_scheduler` should be a '
                            f'dict or a list, but got {_param_scheduler} with '
                            f'type {type(_ParamScheduler)}')

        else:
            raise TypeError(
                '`param_scheduler` should be a `_ParamScheduler`, `dict`, '
                f'list or a tuple, but got {type(param_scheduler)}. If '
                '`param_scheduler` is a list of dict, it means a list of '
                'scheduler configs for single optimizer. If it is a dict and '
                'contains key `type`, it means a scheduler config for a '
                'single optimizer. If it does not contain key `type`, it '
                'means multiple lists of schedulers for multiple optimizers.')

    def _log_env(self, env_cfg: dict) -> None:
        """Logging environment information of the current task.

        Args:
            env_cfg (dict): The environment config of the runner.
        """
        # Collect and log environment information.
        env = collect_env()
        runtime_env = OrderedDict()
        runtime_env.update(env_cfg)
        runtime_env.update(self._randomness_cfg)
        runtime_env['seed'] = self._seed
        runtime_env['Distributed launcher'] = self._launcher
        runtime_env['Distributed training'] = self._distributed
        runtime_env['GPU number'] = self._world_size

        env_info = '\n    ' + '\n    '.join(f'{k}: {v}'
                                            for k, v in env.items())
        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        self.logger.info('\n' + dash_line + '\nSystem environment:' +
                         env_info + '\n'
                         '\nRuntime environment:' + runtime_env_info + '\n' +
                         dash_line + '\n')

        if self.cfg._cfg_dict:
            self.logger.info(f'Config:\n{self.cfg.pretty_text}')

    def _maybe_compile(self, target: str) -> None:
        """Use `torch.compile` to optimize model/wrapped_model."""
        compile_cfg = self.cfg.get('compile', None)
        if compile_cfg is None:
            # no compile options given, won't compile
            return

        if isinstance(compile_cfg, bool):
            if not compile_cfg:
                # compile=False, compilation is disabled
                return
            # compile=True, use default configurations
            compile_cfg = dict()

        assert digit_version(TORCH_VERSION) >= digit_version('2.0.0'), (
            'PyTorch >= 2.0.0 is required to enable torch.compile')
        assert isinstance(compile_cfg, dict), (
            f'`compile` should be a dict or bool, got {type(compile_cfg)}')

        func = getattr(self.model, target)
        compiled_func = torch.compile(func, **compile_cfg)
        setattr(self.model, target, compiled_func)
        self.logger.info('Model has been "compiled". The first few iterations'
                         ' will be slow, please be patient.')
