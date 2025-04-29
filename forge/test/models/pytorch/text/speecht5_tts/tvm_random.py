# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import tvm
from tvm import relay
from tvm.relay.frontend.common import infer_shape


def tvm_bernoulli(x):
    key = tvm.relay.random.threefry_key(42)  # Using a fixed seed
    shape = infer_shape(x)
    rand_normal = relay.random.uniform(key=key, shape=shape, dtype="float32")
    _, sampled_tensor = relay.TupleWrapper(rand_normal, 2)
    output = tvm.relay.cast(tvm.relay.less(sampled_tensor, x), dtype="float32")
    return output


seed = 42
torch.manual_seed(seed)
x_torch = torch.randn(1, 10)
lin = torch.nn.Linear(10, 10)
x_torch = torch.sigmoid(lin(x_torch))
x_np = x_torch.detach().numpy()

x_shape = list(x_np.shape)

x_tvm = tvm.nd.array(x_np)

x_var = relay.var("x", shape=x_np.shape, dtype="float32")

output_x = tvm_bernoulli(x_var)

func = relay.Function([x_var], relay.Tuple([output_x]))

mod = tvm.IRModule.from_expr(func)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm")

dev = tvm.cpu()
module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
module.set_input("x", tvm.nd.array(x_np.astype("float32")))

module.run()
out_x = module.get_output(0).asnumpy()

print("Output X Shape:", out_x.shape)
print("Output X:", out_x)

grid_x = torch.bernoulli(x_torch, p=0.5)
print("Torch Meshgrid X:", grid_x.detach().numpy())
print("Torch Meshgrid X Shape:", grid_x.detach().numpy().shape)
print("np.allclose for X:", np.allclose(out_x, grid_x.detach().numpy()))
