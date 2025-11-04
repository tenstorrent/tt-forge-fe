# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Detr model having both object detection and segmentation model
# https://huggingface.co/docs/transformers/en/model_doc/detr

import pytest
import torch
from third_party.tt_forge_models.detr.object_detection.pytorch import (
    ModelLoader as ObjectDetectionLoader,
)
from third_party.tt_forge_models.detr.object_detection.pytorch import (
    ModelVariant as ObjectDetectionVariant,
)
from third_party.tt_forge_models.detr.segmentation.pytorch import (
    ModelLoader as SegmentationLoader,
)
from third_party.tt_forge_models.detr.segmentation.pytorch import (
    ModelVariant as SegmentationVariant,
)

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    ModelPriority,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import VerifyConfig, verify


class DetrWrapper(torch.nn.Module):
    def __init__(self, model, task="detection"):
        super().__init__()
        self.model = model
        assert task in ["detection", "segmentation"], "Task must be 'detection' or 'segmentation'"
        self.task = task

    def forward(self, pixel_values, pixel_mask):
        output = self.model(pixel_values, pixel_mask)
        if self.task == "detection":
            return (output.logits, output.pred_boxes)
        else:
            return (output.logits, output.pred_masks)


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(ObjectDetectionVariant.RESNET_50, marks=[pytest.mark.xfail]),
    ],
)
def test_detr_detection(variant):
    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DETR,
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
        priority=ModelPriority.P1,
    )

    # Load model and inputs
    loader = ObjectDetectionLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model = DetrWrapper(model, task="detection")
    input_dict = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_dict["pixel_values"], input_dict["pixel_mask"]]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and inference
    _, co_out = verify(
        inputs,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.97)),
    )


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            SegmentationVariant.RESNET_50_PANOPTIC,
            marks=[pytest.mark.xfail],
        )
    ],
)
def test_detr_segmentation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.DETR,
        variant=variant,
        task=Task.SEMANTIC_SEGMENTATION,
        source=Source.HUGGINGFACE,
    )

    # Load the model and inputs
    loader = SegmentationLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    framework_model = DetrWrapper(model, task="segmentation")
    input_dict = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = [input_dict["pixel_values"], input_dict["pixel_mask"]]

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        module_name=module_name,
        compiler_cfg=compiler_cfg,
    )

    # Model Verification and Inference
    _, co_out = verify(inputs, framework_model, compiled_model)
