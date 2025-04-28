# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# Detr model having both object detection and segmentation model
# https://huggingface.co/docs/transformers/en/model_doc/detr

import pytest
from transformers import DetrForObjectDetection, DetrForSegmentation
from loguru import logger
import torch
import forge
from forge.forge_property_utils import Framework, Source, Task
from forge.verify.verify import verify, DepricatedVerifyConfig

from test.models.pytorch.vision.detr.utils.image_utils import preprocess_input_data

class DetrDeocoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model.decoder

    def forward(self, inputs_embeds, encoder_hidden_states, object_queries, query_position_embeddings, encoder_attention_mask):
        return self.model(inputs_embeds, None, encoder_hidden_states, encoder_attention_mask, object_queries, query_position_embeddings)
    
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "facebook/detr-resnet-50",
        )
    ],
)
def test_detr_decoder(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="detr",
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P1")

    # Load the model
    framework_model = DetrForObjectDetection.from_pretrained(variant)
    
    inputs_embeds=torch.randn(1,100,256)
    encoder_hidden_states=torch.randn(1,280,256)
    object_queries=torch.randn(1,280,256)
    query_position_embeddings=torch.randn(1,100,256)
    encoder_attention_mask=torch.randn(1,280)
    
    inputs = [inputs_embeds, encoder_hidden_states, object_queries, query_position_embeddings, encoder_attention_mask]
    framework_model = DetrDeocoder(framework_model)
    logger.info(f"framework_model = {framework_model}")
    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        verify_cfg=DepricatedVerifyConfig(verify_forge_codegen_vs_framework=True),
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
    
class DetrEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model.encoder

    def forward(self, inputs_embeds, object_queries):
        return self.model(inputs_embeds, None, object_queries)[0]
    
@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "facebook/detr-resnet-50",
        )
    ],
)
def test_detr_encoder(forge_property_recorder, variant):
    # Record Forge Property
    module_name = forge_property_recorder.record_model_properties(
        framework=Framework.PYTORCH,
        model="detr",
        variant=variant,
        task=Task.OBJECT_DETECTION,
        source=Source.HUGGINGFACE,
    )

    # Record Forge Property
    forge_property_recorder.record_group("red")
    forge_property_recorder.record_priority("P1")

    # Load the model
    framework_model = DetrForObjectDetection.from_pretrained(variant)
    
    inputs_embeds=torch.randn(1,280,256)
    object_queries=torch.randn(1,280,256)
    
    inputs = [inputs_embeds, object_queries]
    framework_model = DetrEncoder(framework_model)
    logger.info(f"framework_model = {framework_model}")
    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
        verify_cfg=DepricatedVerifyConfig(verify_forge_codegen_vs_framework=True),
        forge_property_handler=forge_property_recorder,
    )

    # Model Verification
    verify(inputs, framework_model, compiled_model, forge_property_handler=forge_property_recorder)
