"""
Test and verification script for the transpiler.
"""
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.checker
import torch
import numpy as np

from .engine import ONNXToForgeTranspiler
from .codegen import generate_forge_module

def verify_model():
    """
    Creates a model, transpiles it, runs it via Torch (inside TIRGraph), 
    and generates the Forge Module code.
    """
    # 1. Create Dummy Model
    # Input X (Float): [1, 1, 4, 4] 
    X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 1, 4, 4])
    # Output Y (Float): [1, 1, 2, 2]
    Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 1, 2, 2])
    
    # Initializer W (Float): [1, 1, 3, 3] 
    W_vals = np.ones((1, 1, 3, 3), dtype=np.float32)
    W_init = onnx.numpy_helper.from_array(W_vals, name='W')
    W_input = onnx.helper.make_tensor_value_info('W', onnx.TensorProto.FLOAT, [1, 1, 3, 3])

    node_conv = onnx.helper.make_node('Conv', ['X', 'W'], ['conv_out'], kernel_shape=[3, 3], pads=[0, 0, 0, 0], name="conv1")
    node_relu = onnx.helper.make_node('Relu', ['conv_out'], ['Y'], name="relu1")
    
    graph_def = onnx.helper.make_graph([node_conv, node_relu], 'test_conv_graph', [X, W_input], [Y], [W_init])
    model_def = onnx.helper.make_model(graph_def, producer_name='forge-test')
    onnx.checker.check_model(model_def)
    
    # 2. Transpile (with debug mode enabled)
    transpiler = ONNXToForgeTranspiler(debug=True)
    tir_graph = transpiler.transpile(model_def)
    
    # 3. Execute
    print("--- Executing Graph ---")
    input_data_np = np.random.randn(1, 1, 4, 4).astype(np.float32)
    input_dict = {'X': torch.from_numpy(input_data_np)} 
    results = tir_graph.run(input_dict)
    print("Execution Output Shape:", results['Y'].shape)

    # 4. Generate Code
    print("\n--- Generated Forge Module ---")
    generated_code = generate_forge_module(tir_graph)
    print(generated_code)


if __name__ == "__main__":
    verify_model()

