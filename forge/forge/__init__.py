# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

# Set home directory paths for forge and forge
def set_home_paths():
    import sys
    import pathlib
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

    # TT_METAL_RUNTIME_ROOT should be one of the following:
    in_wheel_path = forge_path / "forge/tt-metal"
    in_source_path = (forge_path.parent.resolve() / "third_party/tt-mlir/third_party/tt-metal/src/tt-metal").resolve()

    external_source_path = None
    external_ttmlir = os.environ.get("TTMLIR_SOURCE_DIR")
    if not external_ttmlir:
        cmake_cache = forge_path.parent.resolve() / "build/CMakeCache.txt"
        if cmake_cache.exists():
            with cmake_cache.open() as f:
                for line in f:
                    if line.startswith("TTMLIR_SOURCE_DIR:"):
                        cached_path = line.split("=", 1)[1].strip()
                        if cached_path:
                            external_ttmlir = cached_path
                            break

    if external_ttmlir:
        external_source_path = (pathlib.Path(external_ttmlir) / "third_party/tt-metal/src/tt-metal").resolve()

    if "TT_METAL_RUNTIME_ROOT" not in os.environ:
        if in_wheel_path.exists():
            os.environ["TT_METAL_RUNTIME_ROOT"] = str(in_wheel_path)
        elif external_source_path and external_source_path.exists():
            os.environ["TT_METAL_RUNTIME_ROOT"] = str(external_source_path)
        elif in_source_path.exists():
            os.environ["TT_METAL_RUNTIME_ROOT"] = str(in_source_path)
        else:
            tried_paths = [in_wheel_path, in_source_path]
            if external_source_path:
                tried_paths.insert(1, external_source_path)
            logger.error(
                f"TT_METAL_RUNTIME_ROOT environment variable is not set. Tried setting it to {', '.join(str(p) for p in tried_paths)}, but none exist. Something is wrong with the installation."
            )
            exit(1)

    # Check whether we're running from a wheel or from source
    valid_source_paths = [in_source_path]
    if external_source_path:
        valid_source_paths.append(external_source_path)

    if in_wheel_path.exists():
        os.environ["FORGE_IN_WHEEL"] = "1"
    elif any(p.exists() for p in valid_source_paths):
        os.environ["FORGE_IN_SOURCE"] = "1"
    else:
        logger.error("Neither wheel nor source path exist for tt-metal. Please check your installation.")

    all_valid_paths = [in_wheel_path] + valid_source_paths
    if pathlib.Path(os.environ["TT_METAL_RUNTIME_ROOT"]) not in all_valid_paths:
        if pathlib.Path(os.environ["TT_METAL_RUNTIME_ROOT"]).exists():
            logger.warning(
                f"TT_METAL_RUNTIME_ROOT environment variable is set to {os.environ['TT_METAL_RUNTIME_ROOT']}, which looks like a non-standard path. Please check if this is intentional. If set incorrectly, it will cause issues during runtime."
            )
        else:
            logger.error(
                f"TT_METAL_RUNTIME_ROOT environment variable is set to {os.environ['TT_METAL_RUNTIME_ROOT']}, which does not exist. Please check if this is intentional. Unset it so that the default path is used or set it to the correct path."
            )
            exit(1)


set_home_paths()

# eliminate tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from .module import Module, PyTorchModule, ForgeModule, TFGraphDefModule, OnnxModule, JaxModule, TFLiteModule
from .compiled_graph_state import CompiledGraphState
from .config import (
    CompilerConfig,
    CompileDepth,
)
from .verify import DeprecatedVerifyConfig
from .forgeglobal import set_device_pipeline, is_silicon, get_tenstorrent_device
from .parameter import Parameter
from .tensor import Tensor, SomeTensor, TensorShape
from .optimizers import SGD, Adam, AdamW
from ._C import DataFormat, MathFidelity
from ._C import k_dim

import forge.op as op
import forge.transformers

from .compile import compile_main as compile

# Torch backend registration
# TODO: move this in a separate file / module.
from torch._dynamo.backends.registry import _BACKENDS
from torch._dynamo import register_backend
