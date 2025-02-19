import pytest 
import torch
import torch.nn as nn 
from loguru import logger
import tvm.relay as relay
import numpy as np
import tvm 
import scipy.stats 

@pytest.mark.parametrize(
    "shape, mask_shape",
    [
        # ((300,9),(300,)),
        ((10,9),(10,))
    ],
)
def test_index_issue(shape,mask_shape):
    
    class index_issue(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,final_box_preds, mask ):
            boxes3d = final_box_preds[mask]
            return boxes3d
 
    input_tensor = torch.rand(shape)
    mask = torch.randint(0, 2, mask_shape, dtype=torch.bool)
    
    logger.info("our inputs")
    logger.info("input_tensor={}",input_tensor)
    logger.info("mask={}",mask)
    logger.info("***************")
    
    framework_model = index_issue()
    cpu_output = framework_model(input_tensor,mask)
    
    # Convert input data to numpy arrays for TVM
    x_np = input_tensor.numpy()
    mask_np = mask.numpy()
    
    m=np.argwhere(mask_np)
    op_np =np.take(x_np,m,axis=0)
    
    torch.set_printoptions(linewidth=1000,edgeitems=10,precision =20)
    
    
    op_np_cpu = torch.from_numpy(op_np)
    
    
    logger.info("cpu_output.shape={}",cpu_output.shape)
    logger.info("op_np_cpu.shape={}",op_np_cpu.shape)
    
    op_np_cpu = op_np_cpu.squeeze(1)
    
    logger.info("crted op_np_cpu.shape={}",op_np_cpu.shape)
    
    for pt, on in zip(cpu_output, op_np_cpu):
        correlation_coefficient, _ = scipy.stats.pearsonr(pt.detach().numpy().reshape(-1), on.reshape(-1))
        print("PCC = ", correlation_coefficient)

    if torch.allclose(cpu_output, op_np_cpu):
        print("Test passed!")
        print("CPU Output:", cpu_output)
        print("numpy cpu Output:", op_np_cpu)
    else:
        print("Test failed!")
        print("CPU Output:", cpu_output)
        print("numpy cpu Output:", op_np_cpu)
    

    # Define Relay variables
    x1 = relay.var("x", shape=x_np.shape, dtype="float32")
    mask1 = relay.var("mask", shape=mask_np.shape, dtype="bool")
    
    "add tvm logic"
     # Convert NumPy mask to Relay constant
    mask_const = relay.const(mask_np, dtype="bool")

    # Compute indices using relay.argwhere
    mask_indices = relay.argwhere(mask_const)

    # Gather elements using relay.take
    result = relay.take(x1, mask_indices, axis=0)
        
    # Create Relay function
    func = relay.Function([x1, mask1], result)

    # Create the Relay module
    mod = tvm.IRModule.from_expr(func)

    # Build the module
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(mod, target="llvm")

    # Create the TVM GraphExecutor module
    dev = tvm.cpu()
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # Set input and run the module
    module.set_input("x", tvm.nd.array(x_np.astype("float32")))
    module.set_input("mask", tvm.nd.array(mask_np.astype("bool")))
    module.run()

    # Get the output from TVM and compare
    out_tvm = module.get_output(0).asnumpy()

    # Compare the outputs
    op2 = torch.from_numpy(out_tvm)
    if torch.allclose(cpu_output, op2):
        print("Test passed!")
        print("CPU Output:", cpu_output)
        print("TVM Output:", op2)
    else:
        print("Test failed!")
        print("CPU Output:", cpu_output)
        print("TVM Output:", op2)