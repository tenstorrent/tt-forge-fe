# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import forge
from forge.forge_property_utils import (
    Framework,
    ModelArch,
    ModelGroup,
    Source,
    Task,
    record_model_properties,
)
from forge.verify.verify import verify
from loguru import logger

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder_attn = model.model.decoder.layers[0].self_attn

    def forward(self, hidden_states, cache_position):
        
        
        logger.info("=======================================")
        logger.info("self.decoder_attn={}",self.decoder_attn)
        logger.info("hidden_states={}",hidden_states)
        logger.info("cache_position={}",cache_position)
        
        logger.info("hidden_states.dtype={}",hidden_states.dtype)
        logger.info("cache_position.dtype={}",cache_position.dtype)
        
        logger.info("hidden_states.shape={}",hidden_states.shape)
        logger.info("cache_position.shape={}",cache_position.shape)
        

        logger.info("=======================================")
        
        op,_,_ = self.decoder_attn(
            hidden_states=hidden_states,
            cache_position=cache_position
        )
        return op


@pytest.mark.parametrize("variant", ["openai/whisper-large-v3-turbo"])
def test_whisper_large_v3_speech_translation(variant):

    # Record Forge Property
    module_name = record_model_properties(
        framework=Framework.PYTORCH,
        model=ModelArch.WHISPER,
        variant=variant,
        task=Task.SPEECH_TRANSLATE,
        source=Source.HUGGINGFACE,
        group=ModelGroup.RED,
    )

    framework_model = WhisperForConditionalGeneration.from_pretrained(variant,return_dict=False,use_cache=False)
    framework_model = Wrapper(framework_model)
    
    logger.info("framework_model={}",framework_model)
    
    inputs = [torch.load('hidden_states.pt'),torch.load('cache_position.pt')]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, module_name=module_name)

    # Model Verification
    verify(inputs, framework_model, compiled_model)
