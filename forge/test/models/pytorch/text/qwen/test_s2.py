import torch
import torch.nn as nn
import forge
from forge.verify.verify import verify
from test.utils import download_model
from transformers import AutoModel,AutoTokenizer
from loguru import logger
from test.models.models_utils import get_detailed_instruct

def test_linear():
    class linear(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.gate_proj = model.layers[0].mlp.gate_proj
            
        def forward(self,input_ids ):
            logger.info("input_ids={}",input_ids)
            logger.info("input_ids.dtype={}",input_ids.dtype)
            logger.info("input_ids.shape={}",input_ids.shape)
            logger.info("self.gate_proj={}",self.gate_proj)

            inputs_embeds = self.gate_proj(input_ids)
            return inputs_embeds


    variant = "Qwen/Qwen3-Embedding-0.6B"

    framework_model = download_model(AutoModel.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()
    
    has_trainable_params = False
    for name, param in framework_model.named_parameters():
        if param.requires_grad:
            has_trainable_params = True
            logger.info(f"Trainable parameter found: {name}")
            param.requires_grad = False  # Freeze it

    if has_trainable_params:
        logger.info("Trainable parameters found and frozen.")
    else:
        logger.info("No trainable parameters found.")


    model = linear(framework_model)
    model.eval()
    
    inputs = [torch.load('gate_proj_ip.pt')]

    # Compile model
    compiled_model = forge.compile(model, inputs)

    # Model Verification
    verify(inputs, model, compiled_model)
