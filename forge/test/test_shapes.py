# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Testing of various shapes that are not natively supported by Forge
"""

import pytest
import torch
from loguru import logger

import forge
import forge.op
from forge import (
    ForgeModule,
    TTDevice,
    BackendType,
    Tensor,
    forge_compile,
    CompilerConfig,
    VerifyConfig,
)

verify_cfg = VerifyConfig(run_golden=True, run_net2pipe=True, verify_post_placer=False) # Run backend golden check on all tests in here

single_shapes = (

    # Various combinations of 1 and 32
    (1, 1, 1, 1),   
    (1, 1, 32, 1),
    (1, 1, 1, 32),
    (1,),
    (32,),
    (1, 32),
    (32, 1),
    (1, 1, 1, 32, 32),
    (1, 1, 1, 1, 1),

    # not divisible by 32
    (1, 1, 32, 33),
    (1, 1, 33, 32),
    (1, 1, 33, 33),
    (32, 33),
    (33, 32),
    (33, 33),
    (8, 127, 62),

)

# Pairs of compatible shapes where one will be broadcast
broadcast_shapes = (

    # Various combinations of 1 and 32
    ( (1, 1, 32, 1),     (1, 1, 1, 1) ),
    ( (1, 1, 1, 32),     (1, 1, 1, 1) ),
    ( (1,),              (32,) ),
    ( (32,),             (1,) ),
    ( (1, 32),           (1, 1) ),
    ( (32, 1),           (1, 1) ),
    ( (1, 1, 1, 32, 32), (1, 1, 1, 1, 32) ),

    # not divisible by 32
    ( (1, 1, 32, 33),    (1, 1, 1, 33) ),
    ( (1, 1, 33, 32),    (1, 1, 33, 1) ),
    ( (1, 1, 32, 33),    (1, 1, 32, 1) ),
    ( (1, 1, 33, 32),    (1, 1, 1, 32) ),
    ( (1, 1, 33, 33),    (1, 1, 1, 1) ),
    ( (32, 33),          (1, 33) ),
    ( (32, 33),          (32, 1) ),

    # extra dims
    ( (1, 1, 1, 32, 32), (32, 32) ),
    ( (32, 32), (1, 1, 1, 32, 32) ),

)

# Matmul pairs shapes
matmul_shapes = (

    # Various combinations of 1 and 32
    ( (1, 1, 32, 1),     (1, 1, 1, 1) ),
    ( (32, 1),           (1, 1) ),
    ( (1, 32),           (32, 1) ),
    ( (1, 1, 1, 32, 32), (1, 1, 1, 32, 1) ),

    # not divisible by 32
    ( (1, 1, 32, 33),    (1, 1, 33, 32) ),
    ( (1, 1, 33, 32),    (1, 1, 32, 1) ),
    ( (1, 1, 32, 33),    (1, 1, 33, 1) ),
    ( (1, 1, 33, 32),    (1, 1, 32, 32) ),
    ( (1, 1, 33, 33),    (1, 1, 33, 1) ),
    ( (1, 1, 33, 33),    (1, 33, 1) ),
    ( (1, 33, 33),       (33, 1) ),
    ( (32, 33),          (33, 32) ),
    ( (32, 33),          (33, 1) ),

    # extra dims
    ( (1, 1, 1, 32, 32), (32, 32) ),
    ( (32, 32), (1, 1, 1, 32, 32) ),


)

class EltwiseBinary(ForgeModule):
    def __init__(self, name, weight_shape):
        super().__init__(name)
        self.shape = weight_shape
        self.weights1 = forge.Parameter(*self.shape, requires_grad=True)

    def forward(self, act):
        return forge.op.Add("add", act, self.weights1)

class Matmul(ForgeModule):
    def __init__(self, name, weight_shape):
        super().__init__(name)
        self.shape = weight_shape
        self.weights1 = forge.Parameter(*self.shape, requires_grad=True)

    def forward(self, act):
        return forge.op.Matmul("matmul", act, self.weights1)

@pytest.mark.parametrize("shape", single_shapes)
@pytest.mark.parametrize("model", (EltwiseBinary,))
def test_eltwise_binary_same_shape(shape, training, model):


    logger.info(f"Testing shape {shape}")
    mod = model("test_module", shape)
    sgd_optimizer = forge.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = Tensor.create_from_torch(torch.rand(*shape, requires_grad=True))

    mod.set_parameter("weights1", torch.rand(*shape, requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    forge_compile(tt0, "shapes", act, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)

@pytest.mark.parametrize("shapes", broadcast_shapes)
@pytest.mark.parametrize("model", (EltwiseBinary,))
def test_eltwise_binary_broadcast_shapes(shapes, training, model):

    logger.info(f"Testing shapes {shapes}")
    mod = model("test_module", shapes[1])
    sgd_optimizer = forge.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = Tensor.create_from_torch(torch.rand(*shapes[0], requires_grad=True))

    mod.set_parameter("weights1", torch.rand(*shapes[1], requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    forge_compile(tt0, "brcst_shapes", act, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)

    
@pytest.mark.parametrize("shapes", matmul_shapes)
@pytest.mark.parametrize("model", (Matmul,))
def test_matmul_shapes(shapes, training, model):

    logger.info(f"Testing shapes {shapes}")
    mod = model("test_module", shapes[1])
    sgd_optimizer = forge.optimizers.SGD(
        learning_rate=0.5, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = Tensor.create_from_torch(torch.rand(*shapes[0], requires_grad=True))

    mod.set_parameter("weights1", torch.rand(*shapes[1], requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    forge_compile(tt0, "matmul_brcst_shapes", act, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)

def test_tile_broadcast(training):

    # Test simple situation of eltwise add where broadcast within a tile is needed

    class ScalarBroadcast(ForgeModule):
        def __init__(self, name):
            super().__init__(name)
            self.eltwise_param = forge.Parameter(64, 64, requires_grad=True)

        def forward(self, act):

            # (1, 1) + (64, 64) - need to scalar-broadcast act to get correct result
            add = forge.op.Add("add", act, self.eltwise_param)
    
            return add


    mod = ScalarBroadcast("tile_broadcast")
    sgd_optimizer = forge.optimizers.SGD(
        learning_rate=50.0, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = Tensor.create_from_torch(torch.rand((1, 1), requires_grad=True))
    mod.set_parameter("eltwise_param", torch.rand((64, 64), requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    forge_compile(tt0, "tile_broadcast", act, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)
    
def test_tile_fork(training):

    # Test a situation where one side needs a scalar broadcast and the other doesn't (or, in fact, can't have it or would get wrong data)

    class ScalarFork(ForgeModule):
        def __init__(self, name):
            super().__init__(name)
            self.weights1 = forge.Parameter(1, 1, requires_grad=True)
            self.eltwise_param = forge.Parameter(64, 64, requires_grad=True)

        def forward(self, act):

            # (1, 1) x (1, 1) - can't broadcast both inputs or data will be wrong
            mat =  forge.op.Matmul("matmul", act, self.weights1)

            # (1, 1) + (64, 64) - need to scalar-broadcast act to get correct result
            add = forge.op.Add("add", act, self.eltwise_param)
    
            # (1, 1) + (64, 64) - need to scalar broadcast matmul output to get the right result
            return forge.op.Add("final_add", mat, add)


    mod = ScalarFork("tile_fork")
    sgd_optimizer = forge.optimizers.SGD(
        learning_rate=50.0, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = Tensor.create_from_torch(torch.rand((1, 1), requires_grad=True))

    mod.set_parameter("weights1", torch.rand((1, 1), requires_grad=True))
    mod.set_parameter("eltwise_param", torch.rand((64, 64), requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    forge_compile(tt0, "tile_fork", act, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)


def test_reduce_folding(training):

    # Test the scenario where tile broadcast folds into a reduce
    class ReduceFolding(ForgeModule):
        def __init__(self, name):
            super().__init__(name)
            self.eltwise_param = forge.Parameter(64, 64, requires_grad=True)

        def forward(self, act):

            # (64, 64) -> (64, 1)
            red =  forge.op.ReduceSum("reduce", act, dim=-1)

            # (64, 1) + (64, 64) - need to scalar-broadcast act to get correct result
            add = forge.op.Add("add", red, self.eltwise_param)
            return add
    


    mod = ReduceFolding("reduce_folding")
    sgd_optimizer = forge.optimizers.SGD(
        learning_rate=50.0, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = Tensor.create_from_torch(torch.rand((64, 64), requires_grad=True))
    mod.set_parameter("eltwise_param", torch.rand((64, 64), requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    forge_compile(tt0, "reduce_folding", act, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)

def test_input_folding(training):

    # Test the scenario where tile broadcast folds into an input node
    class InputFolding(ForgeModule):
        def __init__(self, name):
            super().__init__(name)
            self.eltwise_param = forge.Parameter(64, 64, requires_grad=True)

        def forward(self, act):

            # (1, 1) + (64, 64) - need to scalar-broadcast act to get correct result
            add = forge.op.Add("add", act, self.eltwise_param)
            return add
    


    mod = InputFolding("input_folding")
    sgd_optimizer = forge.optimizers.SGD(
        learning_rate=50.0, parameters=mod.get_parameters()
    )
    tt0 = TTDevice("tt0", devtype=BackendType.Golden, optimizer=sgd_optimizer)
    tt0.place_module(mod)

    act = Tensor.create_from_torch(torch.rand((1, 1), requires_grad=True))
    mod.set_parameter("eltwise_param", torch.rand((64, 64), requires_grad=True))
    sgd_optimizer.set_optimizer_parameters()

    forge_compile(tt0, "input_folding", act, compiler_cfg=CompilerConfig(enable_training=training), verify_cfg=verify_cfg)

