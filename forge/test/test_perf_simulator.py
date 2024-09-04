# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import forge
from forge.verify import verify_module, VerifyConfig, TestKind
from forge import PyTorchModule
from transformers import BertModel, BertConfig
from test.utils import download_model


def test_bert_encoder():
    forge.config._get_global_compiler_config().compile_depth = forge.config.CompileDepth.GENERATE_NETLIST
        
    model_name = "bert-base-uncased"
    seq_len = 128

    config = download_model(BertConfig.from_pretrained, model_name)
    config.num_hidden_layers = 1

    model = BertModel(config=config)
    encoder = PyTorchModule("bert_encoder", model.encoder)
    microbatch = 1

    os.environ["FORGE_PERF_SIMULATOR"] = "1"
    try:
        verify_module(encoder, [(microbatch, seq_len, config.hidden_size), (microbatch, 1, seq_len, seq_len)],
                VerifyConfig(test_kind=TestKind.INFERENCE, skip_shutdown=True, fp32_fallback=forge.DataFormat.Bfp8_b))
    
        perf_results = forge.forgeglobal.get_devices()[0]._compile_output.perf_model_results
        print(perf_results)
    
    finally:
        del os.environ["FORGE_PERF_SIMULATOR"]

class LayernormFork(forge.ForgeModule):
    """
    Module with a layernorm, and some matmuls
    """

    shape = (1, 1, 128, 512)

    def __init__(self, name):
        super().__init__(name)
        #self.weights1 = forge.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        #self.weights2 = forge.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.ln_weights = forge.Parameter(1, self.shape[-1], requires_grad=True)
        self.ln_bias = forge.Parameter(1, self.shape[-1], requires_grad=True)

    def forward(self, act1):
        #a1 = forge.op.Matmul("matmul1", act1, self.weights1)
        a1 = act1
        a2 = forge.op.Layernorm("layernorm", a1, self.ln_weights, self.ln_bias)
        #a3 = forge.op.Matmul("matmul2", a2, self.weights2)
        a3 = a2
        return a3

def test_layernorm_fork(test_device):
    #forge.config._get_global_compiler_config().compile_depth = forge.config.CompileDepth.GENERATE_NETLIST
        
    microbatch = 64

    os.environ["FORGE_PERF_SIMULATOR"] = "1"
    try:
        forge.config.set_configuration_options(performance_trace=forge.PerfTraceLevel.VERBOSE)
        verify_module(LayernormFork("layernorm_fork"), [(microbatch, LayernormFork.shape[-2], LayernormFork.shape[-1])],
                VerifyConfig(
                    test_kind=TestKind.INFERENCE, 
                    devtype=test_device.devtype, 
                    arch=test_device.arch, 
                    skip_shutdown=True, 
                    fp32_fallback=forge.DataFormat.Bfp8_b))
    
        perf_results = forge.forgeglobal.get_devices()[0]._compile_output.perf_model_results
        print(perf_results)
    
    finally:
        del os.environ["FORGE_PERF_SIMULATOR"]

class MHALikeFork(forge.ForgeModule):
    """
    Module with a layernorm, and some matmuls
    """

    shape = (1, 1, 128, 768)

    def __init__(self, name):
        super().__init__(name)
        self.weights1 = forge.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.weights2 = forge.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.weights3 = forge.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.weights4 = forge.Parameter(self.shape[-1], self.shape[-1], requires_grad=True)
        self.bias = forge.Parameter(1, self.shape[-1], requires_grad=True)

    def forward(self, act1, act2):
        # don't for out of a queue, since that doesn't need buffering
        in1 = act1 - act2

        # fork
        a1 = forge.op.Matmul("matmul1", in1, self.weights1)
        a2 = forge.op.Matmul("matmul2", in1, self.weights2)
        a3 = forge.op.Matmul("matmul3", in1, self.weights3)

        a23 = a2+a3
        a23 = forge.op.Matmul("matmul_a23_1", a23, self.weights4)
        a23 = forge.op.Matmul("matmul_a23_2", a23, self.weights4)
        a23 = forge.op.Matmul("matmul_a23_3", a23, self.weights4)

        return a1 + a23 # join

def test_mha_fork(test_device):
        
    microbatch = 64
    seq_len = MHALikeFork.shape[-2]
    hidden_size = MHALikeFork.shape[-1]
    os.environ["FORGE_PERF_SIMULATOR"] = "1"
    try:
        forge.config.set_configuration_options(performance_trace=forge.PerfTraceLevel.VERBOSE)
        verify_module(MHALikeFork("mha_like_fork"), [(microbatch, seq_len, hidden_size), (microbatch, seq_len, hidden_size)],
                VerifyConfig(
                    test_kind=TestKind.INFERENCE, 
                    devtype=test_device.devtype, 
                    arch=test_device.arch, 
                    skip_shutdown=True, 
                    fp32_fallback=forge.DataFormat.Bfp8_b))
    
        perf_results = forge.forgeglobal.get_devices()[0]._compile_output.perf_model_results
        print(perf_results)
    
    finally:
        del os.environ["FORGE_PERF_SIMULATOR"]
