import os

import torch
from torch import nn


def test_add():
    class Add(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a + b
        
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    
    framework_model = Add()
    fw_out = framework_model(*inputs)
    
    compiled_model = torch.compile(framework_model, backend="tt")
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]


def test_subtract():
    class Subtract(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a - b
        
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    
    framework_model = Subtract()
    fw_out = framework_model(*inputs)
    
    compiled_model = torch.compile(framework_model, backend="tt")
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]


def test_multiply():
    class Multiply(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return a * b
        
    inputs = [torch.rand(1, 32, 32), torch.rand(1, 32, 32)]
    
    framework_model = Multiply()
    fw_out = framework_model(*inputs)
    
    compiled_model = torch.compile(framework_model, backend="tt")
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]


def test_relu():
    class ReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()

        def forward(self, a):
            return self.relu(a)
        
    inputs = [torch.rand(1, 32)]
    
    framework_model = ReLU()
    fw_out = framework_model(*inputs)
    
    compiled_model = torch.compile(framework_model, backend="tt")
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]


def test_linear():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(20, 30, bias=True)

        def forward(self, a):
            return self.l1(a)
        
    inputs = [torch.rand(1, 128, 20)]
    
    framework_model = Linear()
    fw_out = framework_model(*inputs)
    
    compiled_model = torch.compile(framework_model.to("tt"), backend="tt")
    co_out = compiled_model(*[i.to("tt") for i in inputs])
    
    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]


def test_softmax():
    class Softmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, a):
            return self.softmax(a)
        
    inputs = [torch.rand(1, 128)]
    
    framework_model = Softmax()
    fw_out = framework_model(*inputs)
    
    compiled_model = torch.compile(framework_model, backend="tt")
    co_out = compiled_model(*inputs)
    
    co_out = [co.to("cpu") for co in co_out]
    assert [torch.allclose(fo, co) for fo, co in zip(fw_out, co_out)]
