# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge

## https://github.com/RangiLyu/EfficientNet-Lite/
from test.models.pytorch.vision.efficientnet.utils import src_efficientnet_lite as efflite
import os


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_0_pytorch():
    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite0"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite/weights/efficientnet_lite0.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor, module_name="pt_efficientnet_lite_0")


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_1_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite1"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite1.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)

    compiled_model = forge.compile(model, sample_inputs=img_tensor, module_name="pt_efficientnet_lite_1")


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_2_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite2"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite2.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor, module_name="pt_efficientnet_lite_2")


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_3_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite3"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite3.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor, module_name="pt_efficientnet_lite_3")


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_4_pytorch(test_device):

    # STEP 1: Set Forge configuration parameters
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.compile_depth = forge.CompileDepth.SPLIT_GRAPH

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite4"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite4.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor, module_name="pt_efficientnet_lite_3")
