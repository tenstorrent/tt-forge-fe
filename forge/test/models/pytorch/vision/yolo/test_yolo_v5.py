# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify

from third_party.tt_forge_models.yolov5.pytorch import ModelLoader, ModelVariant  # isort:skip

variants = [
    ModelVariant.YOLOV5N,
    ModelVariant.YOLOV5S,
    ModelVariant.YOLOV5M,
    ModelVariant.YOLOV5L,
    ModelVariant.YOLOV5X,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolov5_320x320(restore_package_versions, variant):

    pcc = 0.99
    if variant == ModelVariant.YOLOV5L:
        pcc = 0.95

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="320x320",
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)

    # Unpack the preprocessed input data returned by `load_inputs`
    # ims       : List of input images
    # n         : Number of input samples
    # files     : List of filenames
    # shape0    : Original image shape
    # shape1    : Inference image shape after preprocessing
    # input_tensor : Batched tensor ready for model inference

    ims, n, files, shape0, shape1, input_tensor = loader.load_inputs(dtype_override=torch.bfloat16, input_size=320)
    inputs = [input_tensor]

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification and inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        verify_cfg=VerifyConfig(value_checker=AutomaticValueChecker(pcc=pcc)),
    )

    # Post-process and display results
    loader.post_process(ims, input_tensor.shape, co_out, framework_model, n, shape0, shape1, files)


variants = [
    ModelVariant.YOLOV5N,
    ModelVariant.YOLOV5S,
    ModelVariant.YOLOV5M,
    ModelVariant.YOLOV5L,
    ModelVariant.YOLOV5X,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolov5_640x640(restore_package_versions, variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="640x640",
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)

    # Unpack the preprocessed input data returned by `load_inputs`
    # ims       : List of input images
    # n         : Number of input samples
    # files     : List of filenames
    # shape0    : Original image shape
    # shape1    : Inference image shape after preprocessing
    # input_tensor : Batched tensor ready for model inference

    ims, n, files, shape0, shape1, input_tensor = loader.load_inputs(dtype_override=torch.bfloat16, input_size=640)
    inputs = [input_tensor]

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post-process and display results
    loader.post_process(ims, input_tensor.shape, co_out, framework_model, n, shape0, shape1, files)


variants = [
    ModelVariant.YOLOV5N,
    ModelVariant.YOLOV5S,
    ModelVariant.YOLOV5M,
    ModelVariant.YOLOV5L,
    ModelVariant.YOLOV5X,
]


@pytest.mark.nightly
@pytest.mark.parametrize("variant", variants)
def test_yolov5_480x480(restore_package_versions, variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="480x480",
    )
    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)

    # Unpack the preprocessed input data returned by `load_inputs`
    # ims       : List of input images
    # n         : Number of input samples
    # files     : List of filenames
    # shape0    : Original image shape
    # shape1    : Inference image shape after preprocessing
    # input_tensor : Batched tensor ready for model inference

    ims, n, files, shape0, shape1, input_tensor = loader.load_inputs(dtype_override=torch.bfloat16, input_size=480)
    inputs = [input_tensor]

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post-process and display results
    loader.post_process(ims, input_tensor.shape, co_out, framework_model, n, shape0, shape1, files)


@pytest.mark.nightly
@pytest.mark.parametrize("variant", [ModelVariant.YOLOV5S])
def test_yolov5_1280x1280(restore_package_versions, variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.YOLOV5,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCH_HUB,
        suffix="1280x1280",
    )

    # Load model and inputs
    loader = ModelLoader(variant=variant)
    framework_model = loader.load_model(dtype_override=torch.bfloat16)

    # Unpack the preprocessed input data returned by `load_inputs`
    # ims       : List of input images
    # n         : Number of input samples
    # files     : List of filenames
    # shape0    : Original image shape
    # shape1    : Inference image shape after preprocessing
    # input_tensor : Batched tensor ready for model inference

    ims, n, files, shape0, shape1, input_tensor = loader.load_inputs(dtype_override=torch.bfloat16, input_size=1280)
    inputs = [input_tensor]

    # Configurations
    compiler_cfg = CompilerConfig()
    compiler_cfg.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name=module_name, compiler_cfg=compiler_cfg
    )

    # Model Verification and inference
    _, co_out = verify(inputs, framework_model, compiled_model)

    # Post-process and display results
    loader.post_process(ims, input_tensor.shape, co_out, framework_model, n, shape0, shape1, files)
