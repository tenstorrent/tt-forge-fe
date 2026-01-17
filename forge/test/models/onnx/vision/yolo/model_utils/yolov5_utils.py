# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import onnx
from typing import Optional, Tuple

from test.utils import fetch_model, yolov5_loader
from third_party.tt_forge_models.tools.utils import get_file

# Constants
BASE_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0"
INPUT_SHAPE = (1, 3, 320, 320)
ONNX_OPSET_VERSION = 17


def _create_inputs() -> list:
    """Create input tensor for YOLOv5 model."""
    return [torch.rand(INPUT_SHAPE)]


def _load_and_validate_onnx(onnx_path: str) -> onnx.ModelProto:
    """Load and validate an ONNX model from file path.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Validated ONNX model
    """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def _export_to_onnx(
    model: torch.nn.Module,
    inputs: list,
    output_path: str,
) -> None:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        inputs: Input tensors for the model
        output_path: Path where ONNX model will be saved
    """
    torch.onnx.export(
        model,
        inputs[0],
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=ONNX_OPSET_VERSION,
    )


def _load_onnx_from_s3(size: str) -> Optional[Tuple[onnx.ModelProto, list]]:
    """Try to load ONNX model directly from S3.

    Args:
        size: Model size variant (n, s, m, l, x)

    Returns:
        Tuple of (onnx_model, inputs) if successful, None otherwise
    """
    onnx_path = f"test_files/onnx/yolov5/yolov5{size}_320x320.onnx"
    try:
        onnx_file = get_file(onnx_path)
        onnx_model = _load_and_validate_onnx(str(onnx_file))
        inputs = _create_inputs()
        return onnx_model, inputs
    except Exception as e:
        print(f"Failed to load ONNX from S3 ({onnx_path}): {e}")
        return None


def _load_pytorch_from_s3_and_convert(size: str, forge_tmp_path: str) -> Optional[Tuple[onnx.ModelProto, list]]:
    """Try to load PyTorch weights from S3 and convert to ONNX.

    Args:
        size: Model size variant (n, s, m, l, x)
        forge_tmp_path: Temporary directory path for saving ONNX model

    Returns:
        Tuple of (onnx_model, inputs) if successful, None otherwise
    """
    weight_path = f"test_files/pytorch/yolov5/yolov5{size}.pt"
    try:
        weight_file = get_file(weight_path)

        model = yolov5_loader(str(weight_file), variant="ultralytics/yolov5")
        if model is None:
            raise RuntimeError("yolov5_loader returned None")

        inputs = _create_inputs()
        onnx_path = f"{forge_tmp_path}/yolov5{size}_320x320.onnx"
        _export_to_onnx(model, inputs, onnx_path)

        onnx_model = _load_and_validate_onnx(onnx_path)
        return onnx_model, inputs
    except Exception as e:
        print(f"Failed to load PyTorch weights from S3 and convert ({weight_path}): {e}")
        return None


def _load_via_torch_hub(size: str, forge_tmp_path: str) -> Tuple[onnx.ModelProto, list]:
    """Load model via torch.hub (fallback method).

    Args:
        size: Model size variant (n, s, m, l, x)
        forge_tmp_path: Temporary directory path for saving ONNX model

    Returns:
        Tuple of (onnx_model, inputs)
    """
    name = f"yolov5{size}"
    model = fetch_model(name, f"{BASE_URL}/{name}.pt", yolov5_loader, variant="ultralytics/yolov5", timeout=60)

    inputs = _create_inputs()
    onnx_path = f"{forge_tmp_path}/yolov5{size}_320x320.onnx"
    _export_to_onnx(model, inputs, onnx_path)

    onnx_model = _load_and_validate_onnx(onnx_path)
    return onnx_model, inputs


def load_model_and_inputs(size: str, forge_tmp_path: str) -> Tuple[onnx.ModelProto, list]:
    """Load YOLOv5 model using 3-tier fallback strategy.

    Strategy:
        1. Try loading ONNX directly from S3
        2. Try loading PyTorch weights from S3 and converting
        3. Fall back to torch.hub.load (current method)

    Args:
        size: Model size variant (n, s, m, l, x)
        forge_tmp_path: Temporary directory path for saving ONNX model

    Returns:
        Tuple of (onnx_model, inputs)
    """
    # Tier 1: Try loading ONNX directly from S3
    result = _load_onnx_from_s3(size)
    if result is not None:
        return result

    # Tier 2: Try loading PyTorch weights from S3 and converting
    result = _load_pytorch_from_s3_and_convert(size, forge_tmp_path)
    if result is not None:
        return result

    # Tier 3: Fall back to torch.hub.load
    print("Falling back to torch.hub.load method")
    return _load_via_torch_hub(size, forge_tmp_path)
