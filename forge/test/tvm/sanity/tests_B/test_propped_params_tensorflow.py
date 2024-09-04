# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
#
# Some basic bring-up tests of tracing functionality
#
from operator import is_
from forge.tensor import consteval_input
from forge.tvm_to_python import generate_forge_module
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from loguru import logger

from transformers import BertConfig, TFBertMainLayer, TFBertForQuestionAnswering, OPTConfig, TFOPTModel, GPT2Config, TFGPT2Model
from transformers.models.bert.modeling_tf_bert import TFBertEncoder
from forge.config import CompileDepth, _get_global_compiler_config
from forge.verify import backend, verify_module
from test.utils import download_model
from forge import (
    TFModule,
    TTDevice,
    BackendType,
    CompilerConfig,
    VerifyConfig,
    optimizers,
    forge_compile,
)

def tensor_equals(a, b):
    if len(a.shape) == len(b.shape) and all(dim_a == dim_b for dim_a, dim_b in zip(a.shape, b.shape)):
        return np.allclose(a, b)
    return False

def tensor_in_list(tensor, lst):
    for other in lst:
        if tensor_equals(tensor, other):
            return True
    return False

def is_a_subset_b(A, B):

    for tensor in A:
        if not tensor_in_list(tensor, B):
            print(f"TENSOR: {tensor} NOT IN B")
            return False
    return True

def are_tensor_sets_disjoint(A, B):
    for tensor in A:
        if tensor_in_list(tensor, B):
            return False
    return True

def set_equal(A, B):
    return is_a_subset_b(A, B) and is_a_subset_b(B, A)

def assert_params(framework_mod, forge_mods, const_propped, no_grad_tensors = []):
    # const_propped = [t.numpy() for t in const_propped]

    grad_not_required_framework = no_grad_tensors
    grad_required_framework = []
    for tensor in framework_mod.weights:
        if tensor.trainable:
            grad_required_framework.append(tensor.numpy())
        else:
            grad_not_required_framework.append(tensor.numpy())

    forge_params = []
    for mod in forge_mods:
        forge_params = forge_params + mod.get_parameters()

    grad_required_forge = []
    grad_not_required_forge = []
    for tensor in forge_params:
        if tensor.requires_grad:
            grad_required_forge.append(tensor.value().detach().numpy())
        else:
            grad_not_required_forge.append(tensor.value().detach().numpy())

    # The requires_grad tensors in the framework should still be requires_grad in forge
    assert is_a_subset_b(const_propped, grad_required_forge)

    # The non requires_grad tensors of the framework should be a subset of the non requires_grad tensors of the forge mod
    assert are_tensor_sets_disjoint(grad_not_required_framework, grad_required_forge)

def test_bert():

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask = {'intermediate/dense'} 

    input_shape = (1, 64, 128)
    config = download_model(BertConfig.from_pretrained, "prajjwal1/bert-tiny")
    config.num_hidden_layers = 1
    
    class Wrapper(tf.keras.Model):
        def __init__(self, config):
            super().__init__()
            self.model = TFBertEncoder(config, name='encoder')
            self.am = tf.zeros((1, 64, 64))
            self.hm = tf.zeros((1, 64, 64))

        def call(self, x):
            return self.model(x, self.am, self.hm, None, None, None, None, False, False, False)
    
    model = Wrapper(config)
    act = tf.random.uniform(input_shape)
    inputs = [act]
    model(*inputs)
    const_propped = model.model.layer[0].intermediate.dense.get_weights()

    mod = TFModule("tf_bert", model)
    forge_mods, _, forge_inputs = generate_forge_module(mod, inputs)

    assert_params(model, forge_mods, const_propped, [model.am, model.hm])

def test_gpt2():

    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True
    compiler_cfg.tvm_constnat_prop_mask={"attn/c_attn/weight", "attn/c_attn/bias"} 

    input_shape = (1, 768)

    config = download_model(GPT2Config.from_pretrained, "gpt2")
    config.num_hidden_layers = 1
    config.use_cache = False
    model = TFGPT2Model(config)

    mod = TFModule("gpt2", model)

    inputs = [tf.random.uniform(input_shape, maxval=768, dtype=tf.int32)]
    model(*inputs)
    
    const_propped = [model.transformer.h[0].attn.c_attn.weight.numpy(), model.transformer.h[0].attn.c_attn.bias.numpy()]

    forge_mods, _, forge_inputs = generate_forge_module(mod, inputs, verify_cfg=VerifyConfig(pcc=0.99))

    new_const_propped = []
    for tensor in const_propped:
        new_const_propped = new_const_propped + tf.split(tensor, 3, -1)

    assert_params(model, forge_mods, new_const_propped)


def test_opt():
    
    compiler_cfg = _get_global_compiler_config()
    compiler_cfg.enable_tvm_constant_prop = True 

    configuration = OPTConfig()
    configuration.num_hidden_layers = 1
    model = TFOPTModel(configuration)

    mod = TFModule("OPT_tf", model)
    
    const_propped = [t.numpy() for t in model.weights]

    input_shape = (1, 768)
    inputs = [tf.random.uniform(input_shape, maxval=input_shape[-1], dtype=tf.int32)]
    forge_mods, _, forge_inputs = generate_forge_module(mod, inputs)

    assert_params(model, forge_mods, const_propped)
