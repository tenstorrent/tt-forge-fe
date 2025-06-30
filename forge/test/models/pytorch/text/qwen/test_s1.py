import torch
import torch.nn as nn
import forge
from forge.verify.verify import verify
from test.utils import download_model
from transformers import AutoModel
from loguru import logger
 
def test_silu():
    class silu(nn.Module):
        def __init__(self,model):
            super().__init__()
            self.model = model.layers[3].mlp 
            logger.info("self.model={}", self.model)
            
        def forward(self,ip ):
            logger.info("ip={}",ip)
            logger.info("ip.shape={}",ip.shape)
            logger.info("ip.dtype={}",ip.dtype)
            op = self.model(ip)
            return op
 

    variant = "Qwen/Qwen3-Embedding-0.6B"
    
    framework_model = download_model(AutoModel.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()
    
    model = silu(framework_model)
    model.eval()
    inputs = [torch.load("mlp_ip.pt")]
 
    # Compile model
    compiled_model = forge.compile(model, inputs)
 
    # Model Verification
    verify(inputs, model, compiled_model)