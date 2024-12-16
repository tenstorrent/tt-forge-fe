import inspect
import logging
import sys
import torch
from collections.abc import Callable
from contextlib import contextmanager
from importlib import import_module
from torch import Tensor
from torch import distributed as dist
from torch import nn as nn
from torch.autograd.function import Function
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union, Mapping, Sequence
from custom.imports import ConfigDict, Config
from custom.registry_base import Registry, build_from_cfg
from rich.console import Console
from rich.table import Table
import threading
import warnings
from collections import OrderedDict
from typing import Type, TypeVar
TORCH_VERSION = torch.__version__

_lock = threading.RLock()
T = TypeVar('T')

def build_runner_from_cfg(cfg: Union[dict, ConfigDict, Config],
                          registry: Registry) -> 'Runner':
    """Build a Runner object.

    Examples:
        >>> from mmengine.registry import Registry, build_runner_from_cfg
        >>> RUNNERS = Registry('runners', build_func=build_runner_from_cfg)
        >>> @RUNNERS.register_module()
        >>> class CustomRunner(Runner):
        >>>     def setup_env(env_cfg):
        >>>         pass
        >>> cfg = dict(runner_type='CustomRunner', ...)
        >>> custom_runner = RUNNERS.build(cfg)

    Args:
        cfg (dict or ConfigDict or Config): Config dict. If "runner_type" key
            exists, it will be used to build a custom runner. Otherwise, it
            will be used to build a default runner.
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        object: The constructed runner object.
    """
    from ..config import Config, ConfigDict
    from ..logging import print_log

    assert isinstance(
        cfg,
        (dict, ConfigDict, Config
         )), f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}'
    assert isinstance(
        registry, Registry), ('registry should be a mmengine.Registry object',
                              f'but got {type(registry)}')

    args = cfg.copy()
    # Runner should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.get('runner_type', 'Runner')
        if isinstance(obj_type, str):
            runner_cls = registry.get(obj_type)
            if runner_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry. '
                    f'Please check whether the value of `{obj_type}` is '
                    'correct or it was registered as expected. More details '
                    'can be found at https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module'  # noqa: E501
                )
        elif inspect.isclass(obj_type):
            runner_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        runner = runner_cls.from_cfg(args)  # type: ignore
        print_log(
            f'An `{runner_cls.__name__}` instance is built from '  # type: ignore # noqa: E501
            'registry, its implementation can be found in'
            f'{runner_cls.__module__}',  # type: ignore
            logger='current',
            level=logging.DEBUG)
        return runner


def build_model_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, 'ConfigDict', 'Config']] = None
) -> 'nn.Module':
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, which is either a config
            dict or a list of config dicts. If cfg is a list, the built
            modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn.Module.
    """
    from custom.base import Sequential
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(_cfg, registry, default_args) for _cfg in cfg
        ]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_scheduler_from_cfg(
    cfg: Union[dict, ConfigDict, Config],
    registry: Registry,
    default_args: Optional[Union[dict, ConfigDict, Config]] = None
) -> '_ParamScheduler':
    """Builds a ``ParamScheduler`` instance from config.

    ``ParamScheduler`` supports building instance by its constructor or
    method ``build_iter_from_epoch``. Therefore, its registry needs a build
    function to handle both cases.

    Args:
        cfg (dict or ConfigDict or Config): Config dictionary. If it contains
            the key ``convert_to_iter_based``, instance will be built by method
            ``convert_to_iter_based``, otherwise instance will be built by its
            constructor.
        registry (:obj:`Registry`): The ``PARAM_SCHEDULERS`` registry.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. It must contain key ``optimizer``. If
            ``convert_to_iter_based`` is defined in ``cfg``, it must
            additionally contain key ``epoch_length``. Defaults to None.

    Returns:
        object: The constructed ``ParamScheduler``.
    """
    assert isinstance(
        cfg,
        (dict, ConfigDict, Config
         )), f'cfg should be a dict, ConfigDict or Config, but got {type(cfg)}'
    assert isinstance(
        registry, Registry), ('registry should be a mmengine.Registry object',
                              f'but got {type(registry)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    scope = args.pop('_scope_', None)
    with registry.switch_scope_and_registry(scope) as registry:
        convert_to_iter = args.pop('convert_to_iter_based', False)
        if convert_to_iter:
            scheduler_type = args.pop('type')
            assert 'epoch_length' in args and args.get('by_epoch', True), (
                'Only epoch-based parameter scheduler can be converted to '
                'iter-based, and `epoch_length` should be set')
            if isinstance(scheduler_type, str):
                scheduler_cls = registry.get(scheduler_type)
                if scheduler_cls is None:
                    raise KeyError(
                        f'{scheduler_type} is not in the {registry.name} '
                        'registry. Please check whether the value of '
                        f'`{scheduler_type}` is correct or it was registered '
                        'as expected. More details can be found at https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module'  # noqa: E501
                    )
            elif inspect.isclass(scheduler_type):
                scheduler_cls = scheduler_type
            else:
                raise TypeError('type must be a str or valid type, but got '
                                f'{type(scheduler_type)}')
            return scheduler_cls.build_iter_from_epoch(  # type: ignore
                **args)
        else:
            args.pop('epoch_length', None)
            return build_from_cfg(args, registry)
        
# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', build_func=build_runner_from_cfg)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry('runner constructor')
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop')
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook')

# manage all kinds of strategies like `NativeStrategy` and `DDPStrategy`
STRATEGIES = Registry('strategy')

# manage data-related modules
DATASETS = Registry('dataset')
DATA_SAMPLERS = Registry('data sampler')
TRANSFORMS = Registry('transform')

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', build_model_from_cfg)
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper')
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry('weight initializer')

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer')
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optim_wrapper')
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry('optimizer wrapper constructor')
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler', build_func=build_scheduler_from_cfg)

# manage all kinds of metrics
METRICS = Registry('metric')
# manage evaluator
EVALUATOR = Registry('evaluator')

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util')

# manage visualizer
VISUALIZERS = Registry('visualizer')
# manage visualizer backend
VISBACKENDS = Registry('vis_backend')

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor')

# manage inferencer
INFERENCERS = Registry('inferencer')

# manage function
FUNCTIONS = Registry('function')

MMENGINE_DATA_SAMPLERS = DATA_SAMPLERS
MMENGINE_DATASETS = DATASETS
MMENGINE_EVALUATOR = EVALUATOR
MMENGINE_HOOKS = HOOKS
MMENGINE_INFERENCERS = INFERENCERS
MMENGINE_LOG_PROCESSORS = LOG_PROCESSORS
MMENGINE_LOOPS = LOOPS
MMENGINE_METRICS = METRICS
MMENGINE_MODEL_WRAPPERS = MODEL_WRAPPERS
MMENGINE_MODELS = MODELS

MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS = OPTIM_WRAPPER_CONSTRUCTORS
MMENGINE_OPTIM_WRAPPERS = OPTIM_WRAPPERS
MMENGINE_OPTIMIZERS = OPTIMIZERS
MMENGINE_PARAM_SCHEDULERS = PARAM_SCHEDULERS

MMENGINE_RUNNER_CONSTRUCTORS = RUNNER_CONSTRUCTORS
MMENGINE_RUNNERS = RUNNERS
TASK_UTILS=MMENGINE_TASK_UTILS = TASK_UTILS
MMENGINE_TRANSFORMS = TRANSFORMS
MMENGINE_VISBACKENDS = VISBACKENDS
MMENGINE_VISUALIZERS = VISUALIZERS
MMENGINE_WEIGHT_INITIALIZERS = WEIGHT_INITIALIZERS

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    # TODO: update the location when mmdet3d has its own runner
    'runner',
    parent=MMENGINE_RUNNERS,
    locations=['mmdet3d.engine'])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    # TODO: update the location when mmdet3d has its own runner
    locations=['mmdet3d.engine'])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    # TODO: update the location when mmdet3d has its own loop
    'loop',
    parent=MMENGINE_LOOPS,
    locations=['mmdet3d.engine'])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['mmdet3d.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['mmdet3d.datasets'])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    # TODO: update the location when mmdet3d has its own data sampler
    locations=['mmdet3d.datasets'])
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmdet3d.datasets.transforms'])

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', parent=MMENGINE_MODELS, locations=['mmdet3d.models'])
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmdet3d.models'])
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmdet3d.models'])

# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    # TODO: update the location when mmdet3d has its own optimizer
    locations=['mmdet3d.engine'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    # TODO: update the location when mmdet3d has its own optimizer
    locations=['mmdet3d.engine'])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    # TODO: update the location when mmdet3d has its own optimizer
    locations=['mmdet3d.engine'])
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    # TODO: update the location when mmdet3d has its own scheduler
    locations=['mmdet3d.engine'])
# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['mmdet3d.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMENGINE_EVALUATOR, locations=['mmdet3d.evaluation'])

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, locations=['mmdet3d.models'])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=['mmdet3d.visualization'])
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS,
    locations=['mmdet3d.visualization'])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS,
    # TODO: update the location when mmdet3d has its own log processor
    locations=['mmdet3d.engine'])

# manage inferencer
INFERENCERS = Registry(
    'inferencer',
    parent=MMENGINE_INFERENCERS,
    locations=['mmdet3d.api.inferencers'])

@FUNCTIONS.register_module()
def pseudo_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    The default behavior of dataloader is to merge a list of samples to form
    a mini-batch of Tensor(s). However, in MMEngine, ``pseudo_collate``
    will not stack tensors to batch tensors, and convert int, float, ndarray to
    tensors.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    Args:
        data_batch (Sequence): Batch of data from dataloader.

    Returns:
        Any: Transversed Data in the same format as the data_itement of
        ``data_batch``.
    """  # noqa: E501
    data_item = data_batch[0]
    data_item_type = type(data_item)
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        return data_item_type(*(pseudo_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [pseudo_collate(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [pseudo_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [pseudo_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: pseudo_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return data_batch


@FUNCTIONS.register_module()
def default_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    Different from :func:`pseudo_collate`, ``default_collate`` will stack
    tensor contained in ``data_batch`` into a batched tensor with the
    first dimension batch size, and then move input tensor to the target
    device.

    Different from ``default_collate`` in pytorch, ``default_collate`` will
    not process ``BaseDataElement``.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    Note:
        ``default_collate`` only accept input tensor with the same shape.

    Args:
        data_batch (Sequence): Data sampled from dataset.

    Returns:
        Any: Data in the same format as the data_itement of ``data_batch``, of which
        tensors have been stacked, and ndarray, int, float have been
        converted to tensors.
    """  # noqa: E501
    data_item = data_batch[0]
    data_item_type = type(data_item)
    from custom.base import BaseDataElement
    if isinstance(data_item, (BaseDataElement, str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named_tuple
        return data_item_type(*(default_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [default_collate(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [default_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: default_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return torch_default_collate(data_batch)
    
class AllReduce(Function):

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce in-place operations is unreliable
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        dist.all_reduce(grad_output, async_op=False)
        return grad_output
    
@MODELS.register_module('naiveSyncBN1d')
class NaiveSyncBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args: list, **kwargs: dict) -> None:
        super(NaiveSyncBatchNorm1d, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): Has shape (N, C) or (N, C, L), where N is
                the batch size, C is the number of features or
                channels, and L is the sequence length

        Returns:
            Tensor: Has shape (N, C) or (N, C, L), same shape as input.
        """
        using_dist = dist.is_available() and dist.is_initialized()
        if (not using_dist) or dist.get_world_size() == 1 \
                or not self.training:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        is_two_dim = input.dim() == 2
        if is_two_dim:
            input = input.unsqueeze(2)

        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2])
        meansqr = torch.mean(input * input, dim=[0, 2])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1)
        bias = bias.reshape(1, -1, 1)
        output = input * scale + bias
        if is_two_dim:
            output = output.squeeze(2)
        return output

@MODELS.register_module('naiveSyncBN2d')
class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args: list, **kwargs: dict) -> None:
        super(NaiveSyncBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            Input (Tensor): Feature has shape (N, C, H, W).

        Returns:
            Tensor: Has shape (N, C, H, W), same shape as input.
        """
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        using_dist = dist.is_available() and dist.is_initialized()
        if (not using_dist) or \
                dist.get_world_size() == 1 or \
                not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias
    
def _get_norm() -> tuple:
    """A wrapper to obtain base classes of normalization layers from PyTorch or
    Parrots."""
    if TORCH_VERSION == 'parrots':
        from parrots.nn.modules.batchnorm import _BatchNorm, _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm2d
    else:
        from torch.nn.modules.batchnorm import _BatchNorm
        from torch.nn.modules.instancenorm import _InstanceNorm
        SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_

_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()

def infer_abbr(class_type):
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'
        
def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        # Switch registry to the target scope. If `norm_layer` cannot be found
        # in the registry, fallback to search `norm_layer` in the
        # mmengine.MODELS.
        with MODELS.switch_scope_and_registry(None) as registry:
            norm_layer = registry.get(layer_type)
        if norm_layer is None:
            raise KeyError(f'Cannot find {norm_layer} in registry under '
                           f'scope name {registry.scope}')
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

MODELS.register_module('Conv1d', module=nn.Conv1d)
MODELS.register_module('Conv2d', module=nn.Conv2d)
MODELS.register_module('Conv3d', module=nn.Conv3d)
MODELS.register_module('Conv', module=nn.Conv2d)

def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `conv_layer` cannot be found
    # in the registry, fallback to search `conv_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        conv_layer = registry.get(layer_type)
    if conv_layer is None:
        raise KeyError(f'Cannot find {conv_layer} in registry under scope '
                       f'name {registry.scope}')
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer

class NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, new_shape: tuple) -> torch.Tensor:
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple:
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None
    
# def obsolete_torch_version(torch_version, version_threshold) -> bool:
#     return torch_version == 'parrots' or torch_version <= version_threshold
from packaging import version
def obsolete_torch_version(torch_version, version_threshold) -> bool:
    if torch_version == 'parrots':
        return True
    
    # Convert the torch version string to a version object
    parsed_version = version.parse(torch_version)

    # Compare the parsed version to the threshold (version_threshold must be a tuple or string that can be compared)
    return parsed_version <= version.parse('.'.join(map(str, version_threshold)))

@MODELS.register_module()
@MODELS.register_module('deconv')
class ConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if obsolete_torch_version(TORCH_VERSION, (1, 4)) and x.numel() == 0:
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-2:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)

def build_upsample_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
              deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        upsample = layer_type
    # Switch registry to the target scope. If `upsample` cannot be found
    # in the registry, fallback to search `upsample` in the
    # mmengine.MODELS.
    else:
        with MODELS.switch_scope_and_registry(None) as registry:
            upsample = registry.get(layer_type)
        if upsample is None:
            raise KeyError(f'Cannot find {upsample} in registry under scope '
                           f'name {registry.scope}')
        if upsample is nn.Upsample:
            cfg_['mode'] = layer_type
    layer = upsample(*args, **kwargs, **cfg_)
    return layer