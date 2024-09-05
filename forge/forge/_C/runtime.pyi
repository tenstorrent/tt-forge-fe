# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

class Binary:
    def __init__(self, *args, **kwargs) -> None: ...
    def get_program_inputs(self, *args, **kwargs): ...
    def get_program_outputs(self, *args, **kwargs): ...

def run_binary(arg0: Binary, arg1: int, arg2: list[torch.Tensor]) -> list[torch.Tensor]: ...
