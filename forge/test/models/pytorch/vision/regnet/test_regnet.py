# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from transformers import AutoImageProcessor, RegNetModel, RegNetForImageClassification
import forge
from forge.verify.verify import verify
from .utils.image_utils import preprocess_input_data


@pytest.mark.nightly
@pytest.mark.xfail(
    reason="RuntimeError: TT_FATAL @ tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp:474: new_volume == old_volume. Tracking similar issue on ResNet tenstorrent/tt-mlir#1574 "
)
@pytest.mark.parametrize("variant", ["facebook/regnet-y-040"])
def test_regnet(variant):

    # Load RegNet model
    framework_model = RegNetModel.from_pretrained("facebook/regnet-y-040")

    # Preprocess the image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = preprocess_input_data(image_url, variant)

    # Compiler test
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )

    verify(inputs, framework_model, compiled_model)


@pytest.mark.nightly
@pytest.mark.xfail(
    reason="RuntimeError: TT_FATAL @ tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp:474: new_volume == old_volume. Tracking similar issue on ResNet tenstorrent/tt-mlir#1574 "
)
@pytest.mark.parametrize("variant", ["facebook/regnet-y-040"])
def test_regnet_img_classification(variant):

    # Load the image processor and the RegNet model
    framework_model = RegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

    # Preprocess the image
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    inputs = preprocess_input_data(image_url, variant)

    # Compiler test
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs, module_name="pt_" + str(variant.split("/")[-1].replace("-", "_"))
    )

    verify(inputs, framework_model, compiled_model)
