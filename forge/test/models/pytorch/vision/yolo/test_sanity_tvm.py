# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import scipy.stats
import torch
import torch.nn.functional as F
import tvm
from loguru import logger
from tvm import relay
from tvm.contrib import graph_executor


def resize_relay(x, size):
    x_var = relay.var("x", shape=x.shape, dtype="float32")

    # Resize2D Relay operator
    resized = relay.image.resize2d(x_var, size=size, method="cubic", cubic_alpha=-0.75)

    func = relay.Function([x_var], resized)
    return func


def test_resize_bicubic():
    # Input: random tensor
    x_torch = torch.randn(1, 192, 50, 83)

    # Resize using PyTorch
    y_torch = F.interpolate(x_torch, size=(32, 42), mode="bicubic", align_corners=False)

    # Convert to numpy
    x_np = x_torch.numpy()
    y_np = y_torch.numpy()

    # Build TVM relay function
    func = resize_relay(x_np, size=(32, 42))
    mod = tvm.IRModule.from_expr(func)

    # Compile with TVM
    target = "llvm"
    dev = tvm.cpu()
    with tvm.transform.PassContext(opt_level=5):
        lib = relay.build(mod, target=target, params={})

    # Run with TVM
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("x", tvm.nd.array(x_np.astype("float32")))
    module.run()
    out = module.get_output(0).asnumpy()

    for pt, on in zip(y_torch, out):
        correlation_coefficient, _ = scipy.stats.pearsonr(pt.detach().numpy().reshape(-1), on.reshape(-1))
        logger.info("pearsonr PCC ={} ", correlation_coefficient)

    logger.info("\nout={}", out)
    logger.info("\ny_np={}", y_np)
