import torch
from torch import nn

import forge
from forge.verify.verify import verify
from loguru import logger
from forge.op.eval.common import compare_with_golden

import pytest
from test.utils import download_model
import os
import forge
import requests

from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltForMaskedLM, ViltConfig
from test.models.pytorch.multimodal.vilt.utils.model import ViLtEmbeddingWrapper, ViltModelWrapper
from forge import Tensor

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text1 = "How many cats are there?"
text2 = "a bunch of cats laying on a [MASK]."



def test_check():
    
    class c(nn.Module):
        def __init__(self):
            super().__init__()
            self.layernorm = nn.LayerNorm(768, eps=1e-12)

        def forward(self, hs):
            
            op = self.layernorm(hs)

            return op
        
    variant = "dandelin/vilt-b32-mlm"
    config = ViltConfig.from_pretrained(variant)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    # Load model and processor from HuggingFace
    processor = download_model(ViltProcessor.from_pretrained, variant)
    model = download_model(ViltForMaskedLM.from_pretrained, variant, config=config)
    model.eval()
    
    # prepare inputs
    encoding = processor(image, text2, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    vilt_model = ViltModelWrapper(model=model, task="maskedlm", text_seq_len=encoding["input_ids"].shape[1])

    embedding_output ,_ = text_vision_embedding_model(**encoding)
    inputs = [embedding_output.detach().cpu()]
    
    print("inputs",inputs)
    

    framework_model = c()
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)
    
    co_out = compiled_model(*inputs)
    fw_out = framework_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
    
    import numpy as np
    torch.set_printoptions(linewidth=1000,edgeitems=10,precision =20)
    
    print("cpu output.shape",co_out[0].shape)
    print("forge output.shape",fw_out[0].shape)
    
    print("=========================================")
    print("cpu output\n",co_out)
    print("\nforge output\n",fw_out)
    print("=========================================")


    verify(inputs, framework_model, compiled_model)