# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import forge._C
import torch

class TTDevice:
    def __init__(self, *args, **kwargs) -> None: ...
    def dispatch(self, arg0: Workload, arg1: int, arg2: list[torch.Tensor], arg3: bool) -> list[torch.Tensor]: ...
    def str(self) -> str: ...
    def torch_device(self) -> torch.device: ...
    @property
    def arch(self) -> forge._C.Arch: ...
    @property
    def cluster_yaml(self) -> str: ...
    @property
    def input_runtime_transforms(self) -> dict[int, list[str]]: ...
    @property
    def input_tile_bcast_dims(self) -> dict[int, list[list[int]]]: ...
    @property
    def mmio(self) -> bool: ...
    @property
    def output_runtime_transforms(self) -> dict[int, list[str]]: ...

class TTForgeTensorDesc:
    def __init__(self, name: str, shape: list[int], ptr: int = ..., constant: torch.Tensor | None = ...) -> None: ...
    @property
    def constant(self) -> torch.Tensor | None: ...
    @property
    def name(self) -> str: ...
    @property
    def ptr(self) -> int: ...
    @property
    def shape(self) -> list[int]: ...

class Workload:
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def constants(self) -> list[TTForgeTensorDesc]: ...
    @property
    def inputs(self) -> dict[int, list[TTForgeTensorDesc]]: ...
    @property
    def outputs(self) -> dict[int, list[TTForgeTensorDesc]]: ...
    @property
    def parameters(self) -> list[TTForgeTensorDesc]: ...

def get_available_devices(*args, **kwargs): ...
def get_default_device(*args, **kwargs): ...
def is_created_on_device(arg0: torch.Tensor) -> bool: ...
def original_shape(arg0: torch.Tensor) -> list[int]: ...
def unique_id(arg0: torch.Tensor) -> int: ...
