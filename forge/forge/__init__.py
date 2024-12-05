# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Standard Library
import os


# Set home directory paths for forge and forge
def set_home_paths():
    # Standard Library
    import pathlib
    import sys

    # Third Party
    from loguru import logger

    forge_path = pathlib.Path(__file__).parent.parent.resolve()

    # deployment path
    base_path = str(forge_path)
    out_path = "."

    if "FORGE_HOME" not in os.environ:
        os.environ["FORGE_HOME"] = str(forge_path)
    if "TVM_HOME" not in os.environ:
        os.environ["TVM_HOME"] = str(base_path) + "/tvm"
    if "FORGE_OUT" not in os.environ:
        os.environ["FORGE_OUT"] = str(out_path)
    if "LOGGER_FILE" in os.environ:
        sys.stdout = open(os.environ["LOGGER_FILE"], "w")
        logger.remove()
        logger.add(sys.stdout)


set_home_paths()

# eliminate tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Third Party
from torch._dynamo import register_backend

# Torch backend registration
# TODO: move this in a separate file / module.
from torch._dynamo.backends.registry import _BACKENDS

# Local Imports
import forge.op as op
import forge.transformers
import forge.typing

from ._C import DataFormat, MathFidelity, k_dim
from .compile import compile_main as compile
from .compile import forge_compile_torch
from .compiled_graph_state import CompiledGraphState
from .config import (
    CompileDepth,
    CompilerConfig,
    PerfTraceLevel,
    _internal_insert_fj_buffering_nop,
    configure_mixed_precision,
    insert_buffering_nop,
    insert_nop,
    override_dram_queue_placement,
    override_op_size,
    set_chip_break,
    set_configuration_options,
    set_epoch_break,
)
from .forgeglobal import (
    forge_reset,
    get_tenstorrent_device,
    is_silicon,
    set_device_pipeline,
)
from .module import (
    ForgeModule,
    JaxModule,
    Module,
    OnnxModule,
    PyTorchModule,
    TFGraphDefModule,
    TFLiteModule,
)
from .optimizers import LAMB, LARS, SGD, Adam, AdamW
from .parameter import Parameter
from .tensor import SomeTensor, Tensor, TensorShape
from .torch_compile import compile_torch
from .verify import DepricatedVerifyConfig

# register backend with torch:
# - enables backend to be shown when calling torch._dynamo.list_backends()
# - enables torch.compile(model, backend="<name_from_list_backends>"), where <name_from_list_backends> is "tt" in this case
if "tt" in _BACKENDS:
    del _BACKENDS["tt"]
register_backend(compile_torch, "tt")
