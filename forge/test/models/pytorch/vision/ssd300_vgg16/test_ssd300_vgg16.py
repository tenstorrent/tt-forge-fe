# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict
from typing import Optional

import pytest
import torch
from torch import Tensor

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
from forge.verify.verify import verify

from test.models.models_utils import print_cls_results
from test.models.pytorch.vision.ssd300_vgg16.model_utils.model_utils import (
    Postprocessor,
)
from test.models.pytorch.vision.vision_utils.utils import load_vision_model_and_input

variants_with_weights = {
    "ssd300_vgg16": "SSD300_VGG16_Weights",
}


class SSDWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        model.eval()
        self.model = model

    def forward(
        self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
    ) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:

        # transform the input
        images, targets = self.model.transform(images, targets)

        # get the features from the backbone
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the ssd heads outputs using the features
        head_outputs = self.model.head(features)
        output = [head_outputs["bbox_regression"], head_outputs["cls_logits"], features[0]]

        return output


@pytest.mark.nightly
@pytest.mark.xfail
@pytest.mark.parametrize("variant", variants_with_weights.keys())
def test_ssd300_vgg16(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.SSD300VGG16,
        variant=variant,
        task=Task.IMAGE_CLASSIFICATION,
        source=Source.TORCHVISION,
    )

    # Load model and input
    weight_name = variants_with_weights[variant]
    framework_model, inputs = load_vision_model_and_input(variant, "detection", weight_name)
    model = SSDWrapper(framework_model)
    model.to(torch.bfloat16)
    inputs = [inputs[0].to(torch.bfloat16)]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification
    fw_out, co_out = verify(inputs, model, compiled_model)

    # Post Processing
    postprocessor = Postprocessor(model)
    detection_fw, detection_co = postprocessor.process(fw_out, co_out, inputs)

    # Run model on sample data and print results
    print_cls_results(detection_fw[0], detection_co[0])
