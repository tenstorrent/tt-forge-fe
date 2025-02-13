import pytest
import torch
from torch import nn

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from loguru import logger

def test_view():
    class view(nn.Module):
        def __init__(self):
            super().__init__()
           
        def forward(self, ip1,ip2):
            return torch.stack((ip1,ip2),dim=4).view(1, 6, 20, 50, -1)  
        
    pos_n = torch.rand(1, 6, 20, 50, 128)
        
    ip1 = pos_n[:, :, :, :, 0::2]
    ip2 = pos_n[:, :, :, :, 1::2]

    inputs = [ip1,ip2]
    
    framework_model = view()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    