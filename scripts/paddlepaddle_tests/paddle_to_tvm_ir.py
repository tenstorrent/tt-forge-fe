import paddle 
import tvm
from tvm import relay 
import torch
import os

class PaddleDouble(paddle.nn.Layer):
    def __init__(self):
        super(PaddleDouble, self).__init__()

    def forward(self, x):
        return x + x
    
class PyTorchDouble(torch.nn.Module):
    def __init__(self):
        super(PyTorchDouble, self).__init__()

    def forward(self, x):
        return x + x

class PaddleMNISTLinear(paddle.nn.Layer):
    def __init__(
        self, input_size=784, output_size=10, hidden_size=512, bias=True, dtype='float32'
    ): 
        super(PaddleMNISTLinear, self).__init__()
        self.model = paddle.nn.Sequential(
            paddle.nn.Linear(input_size, hidden_size, bias_attr=bias),
            paddle.nn.ReLU(),
            paddle.nn.Linear(hidden_size, hidden_size, bias_attr=bias),
            paddle.nn.ReLU(),
            paddle.nn.Linear(hidden_size, output_size, bias_attr=bias)
        )
        

    def forward(self, x):
        logits = self.model(x)
        return logits
class PyTorchMNISTLinear(torch.nn.Module):
    def __init__(
        self, input_size=784, output_size=10, hidden_size=512, bias=True, dtype=torch.float32
    ):  
        super(PyTorchMNISTLinear, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=bias),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=bias),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
  

def construct_from_paddle(paddle_model, input_shape):
    input_spec = [paddle.static.InputSpec(shape=input_shape, dtype='float32')]
    traced_model = paddle.jit.to_static(paddle_model, input_spec=input_spec, full_graph=True)

    # hope there is a better way to do this
    # must be paddle.jit.TranslatedLayer or paddle.static.Program
    # paddle.static.save_inference_model and load_inference_model could be used if needed
    model_save_path = "traced_model"
    paddle.jit.save(traced_model, model_save_path)
    loaded_model = paddle.jit.load("traced_model")

    # clean up
    for ext in [".pdiparams", ".pdiparams.info", ".pdmodel"]:
        file = f"{model_save_path}{ext}"
        if os.path.exists(file):
            os.remove(file)

    # input names must match with the ones in the forward method
    inputs = {"x": input_shape} 
    tvm_mod, _ = tvm.relay.frontend.from_paddle(loaded_model, inputs) # no spans

    tvm_mod = tvm.IRModule.from_expr(
                tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], {})
            )
    
    return tvm_mod

def construct_from_pytorch(pytorch_model, input_shape):
    traced_model = torch.jit.trace(pytorch_model, torch.randn(input_shape, dtype=torch.float32))

    # unlike for paddle, input names are not important here
    inputs = [('x', input_shape)]
    tvm_mod, _ = tvm.relay.frontend.from_pytorch(traced_model, inputs)

    tvm_mod = tvm.IRModule.from_expr(
                    tvm.relay.build_module.bind_params_by_name(tvm_mod["main"], {})
                )
    
    return tvm_mod

def construct_and_compare_modules(paddle_model, pytorch_model, input_shape):
    tvm_mod_paddle = construct_from_paddle(paddle_model, input_shape)
    tvm_mod_pytorch = construct_from_pytorch(pytorch_model, input_shape)

    print("TVM module from Paddle: \n\n", tvm_mod_paddle)
    print("TVM module from PyTorch: \n\n", tvm_mod_pytorch)
    print("Are the TVM modules from Paddle and PyTorch equivalent? ", tvm.ir.structural_equal(tvm_mod_paddle, tvm_mod_pytorch))

def test_mnist():
    paddle_model = PaddleMNISTLinear()
    pytorch_model = PyTorchMNISTLinear()
    input_shape = (1, 784)
    construct_and_compare_modules(paddle_model, pytorch_model, input_shape)

def test_double():
    paddle_model = PaddleDouble()
    pytorch_model = PyTorchDouble()
    input_shape = (24, 24)
    construct_and_compare_modules(paddle_model, pytorch_model, input_shape)

if __name__ == "__main__":
    test_double()
    test_mnist()


    