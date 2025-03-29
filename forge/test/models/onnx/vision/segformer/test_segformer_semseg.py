# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from transformers import AutoImageProcessor
import os
import pytest
import requests
from PIL import Image
import onnx

# TODO: These are old forge, we should update them to the currently version.
# import forge
# from forge.verify.backend import verify_module
# from forge import DepricatedVerifyConfig
# from forge.verify.config import TestKind


def get_sample_data(model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    return pixel_values


variants_semseg = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b1-finetuned-ade-512-512",
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b3-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
]


@pytest.mark.skip_model_analysis
@pytest.mark.skip(reason="Requires restructuring")
@pytest.mark.parametrize("variant", variants_semseg)
@pytest.mark.nightly
def test_segformer_semantic_segmentation_onnx(test_device, variant):

    # Set Forge configuration parameters
    compiler_cfg = forge.config.CompilerConfig()
    compiler_cfg.default_df_override = forge.DataFormat.Float16_b
    pcc_value = 0.99

    if test_device.arch == forge.BackendDevice.Wormhole_B0:
        if variant in [
            "nvidia/segformer-b1-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-ade-512-512",
            "nvidia/segformer-b3-finetuned-ade-512-512",
            "nvidia/segformer-b4-finetuned-ade-512-512",
        ]:

            os.environ["FORGE_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

        if variant == "nvidia/segformer-b2-finetuned-ade-512-512" and test_device.devtype == forge.BackendType.Silicon:
            pcc_value = 0.98

    elif test_device.arch == forge.BackendDevice.Grayskull:
        compiler_cfg.enable_auto_fusing = False

        if variant == "nvidia/segformer-b2-finetuned-ade-512-512":
            compiler_cfg.place_on_new_epoch("add_1423")
            compiler_cfg.place_on_new_epoch("concatenate_1427.dc.concatenate.0")

        if variant == "nvidia/segformer-b3-finetuned-ade-512-512":
            compiler_cfg.place_on_new_epoch("add_2431")
            compiler_cfg.place_on_new_epoch("concatenate_2435.dc.concatenate.0")

        if variant == "nvidia/segformer-b4-finetuned-ade-512-512":
            compiler_cfg.place_on_new_epoch("add_3523")
            compiler_cfg.place_on_new_epoch("concatenate_3527.dc.concatenate.0")

        if test_device.devtype == forge.BackendType.Silicon:

            if variant in [
                "nvidia/segformer-b0-finetuned-ade-512-512",
                "nvidia/segformer-b2-finetuned-ade-512-512",
                "nvidia/segformer-b4-finetuned-ade-512-512",
            ]:
                pcc_value = 0.98

            if variant == "nvidia/segformer-b1-finetuned-ade-512-512":
                pcc_value = 0.97

    # Load the sample image
    pixel_values = get_sample_data(variant)

    onnx_model_path = (
        "third_party/confidential_customer_models/generated/files/"
        + str(variant).split("/")[-1].replace("-", "_")
        + ".onnx"
    )
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)

    tt_model = forge.OnnxModule(str(variant).split("/")[-1].replace("-", "_"), model)

    # Run inference on Tenstorrent device
    verify_module(
        tt_model,
        input_shapes=[(pixel_values.shape,)],
        inputs=[(pixel_values,)],
        verify_cfg=DepricatedVerifyConfig(
            arch=test_device.arch,
            devtype=test_device.devtype,
            devmode=test_device.devmode,
            test_kind=TestKind.INFERENCE,
            verify_forge_codegen_vs_framework=True,
            verify_tvm_compile=True,
            pcc=pcc_value,
        ),
    )
