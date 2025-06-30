import torch
import torch.nn as nn
import forge
from forge.verify.verify import verify
from transformers.activations import ACT2FN
from test.utils import download_model
from transformers import AutoModel
from loguru import logger
 
def test_silu():
    class silu(nn.Module):
        def __init__(self,act):
            super().__init__()
            self.act_fn = ACT2FN[act]
            
        def forward(self,ip ):
            logger.info("ip={}",ip)
            logger.info("ip.shape={}",ip.shape)
            logger.info("ip.dtype={}",ip.dtype)
            logger.info("self.act_fn={}",self.act_fn)
            op = self.act_fn(ip)
            return op
 

    variant = "Qwen/Qwen3-Embedding-0.6B"
    
    framework_model = download_model(AutoModel.from_pretrained, variant, return_dict=False, use_cache=False)
    framework_model.eval()
    
    model = silu(framework_model.config.hidden_act)
    model.eval()
    inputs = [torch.load('silu_ip.pt')]
 
    # Compile model
    compiled_model = forge.compile(model, inputs)
 
    # Model Verification
    verify(inputs, model, compiled_model)