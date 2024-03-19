# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import pytest
from tensorflow.security.fuzzing import py
import torch
from pybuda.torch_compile import compile_torch
import copy

class NoOutputGraph(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        a = 3 * a

def test_no_output_graph():
    model = torch.compile(NoOutputGraph(), backend=compile_torch)
    input = torch.tensor([[1.0]])
    tt_res = model(input.to('tt'))

    cpu_res = NoOutputGraph()(input)
    assert cpu_res == tt_res

class DanglingOps(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        b = a + 2
        c = b * 12
        return a

def test_dangling_ops():
    model = torch.compile(DanglingOps(), backend=compile_torch)
    input = torch.tensor([[1.0]])
    tt_res = model(input.to('tt'))
    tt_res = tt_res.to('cpu')

    cpu_res = DanglingOps()(input)
    assert cpu_res == tt_res

foobar = 5.0
class DisjointedGraphs(torch.nn.Module):
    def forward(self, a):
        a = a + 1
        a = a.to('cpu')
        if a[0] > foobar:
            b = a + 2
        else:
            b = a + 3

        return b

def test_disjointed_graphs():
    model = torch.compile(DisjointedGraphs(), backend=compile_torch)
    input = torch.tensor([[1.0]])
    tt_res_ = model(input.to('tt'))
    tt_res_ = tt_res_.to('cpu')
    tt_res = model(input.to('tt'))
    tt_res = tt_res.to('cpu')
    cpu_res = DisjointedGraphs()(input)
    assert cpu_res == tt_res

    input = torch.tensor([[2.5]])
    tt_res = model(input.to('tt'))
    tt_res = tt_res.to('cpu')

    cpu_res = DisjointedGraphs()(input)
    assert cpu_res == tt_res

@pytest.mark.skip(reason="https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2428")
def test_to_double():
    tensor = torch.rand(32, 32).to('tt')
    tensor.to(dtype=torch.double)
 
@pytest.mark.skip(reason="https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2429")
def test_print():
    tensor = torch.rand(32, 32).to('tt')
    print(tensor)

@pytest.mark.skip(reason="https://yyz-gitlab.local.tenstorrent.com/tenstorrent/pybuda/-/issues/2438")
def test_longint():
    original_data = torch.randint(0, 10, (1, 8))
    tensor = original_data.to('tt').to(dtype=torch.int).to('cpu')

    original_data = original_data.to(dtype=torch.int)
    assert torch.allclose(original_data, tensor)

class DisjointedGraphsWithParams(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1, bias=False)
        self.linear2 = torch.nn.Linear(1, 1, bias=False)
    def forward(self, a):
        a = self.linear1(a)
        a = a.to('cpu')
        if a[0] > 1:
            b = a + 2
        else:
            b = self.linear2(a) 

        return b

def test_disjointed_graphs_with_params():
    torch.set_num_threads(1) # TODO: Multi-thread seems to cause data mismatch
    cpu_model = DisjointedGraphsWithParams()
    input = torch.tensor([[1.0]])
    cpu_res = cpu_model(input)
    model = torch.compile(cpu_model.to('tt'), backend=compile_torch)
    tt_res = model(input.to('tt'))

    # Inference
    tt_res = model(input.to('tt'))
    tt_res = tt_res.to('cpu')

    assert cpu_res == tt_res

class NonAlignedSize(torch.nn.Module):
    def forward(self, a):
        return a + 1

@pytest.mark.parametrize("rows", [1, 32])
def test_return_non_aligned_sizes(rows):
    model = torch.compile(NonAlignedSize(), backend=compile_torch)
    input = torch.rand(1, rows, 33)
    input_tt = input.to('tt')
    tt_res = model(input_tt).to('cpu')
    cpu_res = NonAlignedSize()(input)
    assert torch.allclose(cpu_res, tt_res, atol=0, rtol=1e-3)
