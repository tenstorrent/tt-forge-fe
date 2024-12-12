# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Detr model having both object detection and segmentation model
# https://huggingface.co/docs/transformers/en/model_doc/detr

import pytest
from transformers import (
    DetrForObjectDetection,
    DetrForSegmentation,
)
import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from test.models.pytorch.vision.detr.utils.image_utils import preprocess_input_data


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="Tensor mismatch on Framework vs. Forge codegen output 0")
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50"])
def test_detr_detection(variant):

    # Load the model
    compiler_cfg = forge.config._get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.convert_framework_params_to_tvm = True
    framework_model = DetrForObjectDetection.from_pretrained(variant)

    # Preprocess the image for the model
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    input_batch = preprocess_input_data(image_url)

    # Compiler test
    compiled_model = forge.compile(
        framework_model, sample_inputs=[input_batch], module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )

    verify([input_batch], framework_model, compiled_model, VerifyConfig(verify_allclose=False))


@pytest.mark.nightly
@pytest.mark.model_analysis
@pytest.mark.xfail(reason="AssertionError: TVM einsum decomposition does not support bqnc,bnchw->bqnhw yet.")
@pytest.mark.parametrize("variant", ["facebook/detr-resnet-50-panoptic"])
def test_detr_segmentation(variant):
    # Load the model
    framework_model = DetrForSegmentation.from_pretrained(variant)

    # Preprocess the image for the model
    image_url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    input_batch = preprocess_input_data(image_url)

    # since it hangs on error adding xfail here
    pytest.xfail(reason="AssertionError: TVM einsum decomposition does not support bqnc,bnchw->bqnhw yet.")

    # Compiler test
    compiled_model = forge.compile(
        framework_model, sample_inputs=[input_batch], module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )

    verify([input_batch], framework_model, compiled_model, VerifyConfig(verify_allclose=False))
