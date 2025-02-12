import paddle 
import tvm
from tvm import relay # ?
import torch

class PaddleDouble(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x
    
class PyTorchDouble(torch.nn.Module):
    def __init__(self):
        super(PyTorchDouble, self).__init__()

    def forward(self, x):
        return x + x


def construct_from_paddle():
    paddle_model = PaddleDouble()
    input_shape = (24, 24)  

    input_spec = [paddle.static.InputSpec(shape=input_shape, dtype='float32')]
    traced_model = paddle.jit.to_static(paddle_model, input_spec=input_spec)

    # there must be a better way to do this
    model_save_path = "traced_model"
    paddle.jit.save(traced_model, model_save_path)
    loaded_model = paddle.jit.load("traced_model")

    inputs = {"x": input_shape}
    tvm_mod, _ = tvm.relay.frontend.from_paddle(loaded_model, inputs)

    tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], {})
            )
    return tvm_mod

def construct_from_pytorch():
    pytorch_model = PyTorchDouble()
    input_shape = (24, 24)  

    traced_model = torch.jit.trace(pytorch_model, torch.randn(input_shape, dtype=torch.float32))

    inputs = [('x', input_shape)]
    tvm_mod, _ = tvm.relay.frontend.from_pytorch(traced_model, inputs)

    tvm_mod = tvm.IRModule.from_expr(
                    tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], {})
                )
    
    return tvm_mod

if __name__ == "__main__":
    tvm_mod_paddle = construct_from_paddle()
    tvm_mod_pytorch = construct_from_pytorch()

    print("TVM module from Paddle: \n\n", tvm_mod_paddle)
    print("TVM module from PyTorch: \n\n", tvm_mod_pytorch)
    print("Are the TVM modules from Paddle and PyTorch equivalent? ", tvm.ir.structural_equal(tvm_mod_paddle, tvm_mod_pytorch))