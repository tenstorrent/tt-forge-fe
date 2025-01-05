# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
import forge

## https://github.com/RangiLyu/EfficientNet-Lite/
from test.models.pytorch.vision.efficientnet.utils import src_efficientnet_lite as efflite
import os
import torch
from forge.verify.compare import compare_with_golden
from test.models.utils import build_module_name, Framework


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_0_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_0")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite0"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite/weights/efficientnet_lite0.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=[img_tensor], module_name=module_name)
    co_out = compiled_model(img_tensor)
    fw_out = model(img_tensor)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_1_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_1")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite1"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite1.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)

    compiled_model = forge.compile(model, sample_inputs=[img_tensor], module_name=module_name)

    co_out = compiled_model(img_tensor)
    fw_out = model(img_tensor)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_2_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_2")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite2"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite2.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=[img_tensor], module_name=module_name)

    co_out = compiled_model(img_tensor)
    fw_out = model(img_tensor)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_3_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_3")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite3"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite3.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)
    compiled_model = forge.compile(model, sample_inputs=img_tensor, module_name=module_name)

    co_out = compiled_model(img_tensor)
    fw_out = model(img_tensor)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])


@pytest.mark.skip(reason="dependent on CCM repo")
@pytest.mark.nightly
def test_efficientnet_lite_4_pytorch(record_forge_property):
    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_4")

    record_forge_property("module_name", module_name)

    # STEP 2: Model load in Forge
    model_name = "efficientnet_lite4"
    model = efflite.build_efficientnet_lite(model_name, 1000)
    model.load_pretrain("efficientnet_lite4.pth")
    model.eval()

    # Image preprocessing
    wh = efflite.efficientnet_lite_params[model_name][2]
    img_tensor = efflite.get_image_tensor(wh)

    module_name = build_module_name(framework=Framework.PYTORCH, model="efficientnet", variant="lite_4")
    compiled_model = forge.compile(model, sample_inputs=[img_tensor], module_name=module_name)

    co_out = compiled_model(img_tensor)
    fw_out = model(img_tensor)

    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out

    assert all([compare_with_golden(golden=fo, calculated=co, pcc=0.99) for fo, co in zip(fw_out, co_out)])
