# ONNX Transpiler Test Suite

This directory contains comprehensive test cases for the ONNX to TIR transpiler.

## Test Structure

- `test_utils.py`: Utility functions for creating ONNX models programmatically and comparing results
- `test_reduction_ops.py`: Test cases for reduction operations (ReduceSum, ReduceMean, ReduceMax)
- Additional test files for other operations (to be added)

## Running Tests

### Run all reduction operation tests:
```bash
pytest forge/forge/transpiler/frontends/onnx/tests/test_reduction_ops.py -v
```

### Run a specific test:
```bash
pytest forge/forge/transpiler/frontends/onnx/tests/test_reduction_ops.py::TestReduceSum::test_reducesum_basic -v
```

### Run with debug output:
```bash
pytest forge/forge/transpiler/frontends/onnx/tests/test_reduction_ops.py -v -s
```

## Test Coverage

Each test case:
1. Creates an ONNX model programmatically with specific configurations
2. Transpiles the model to TIR with debug mode enabled
3. Verifies the TIR graph structure matches the ONNX model
4. Compares TIR execution outputs with ONNX Runtime outputs
5. Validates correctness across different:
   - Input shapes
   - Data types
   - Opset versions
   - Operation attributes

## Current Test Coverage

### Reduction Operations
- **ReduceSum**: Tests various input shapes, axes, keepdims, and opset versions (1, 11, 13)
- **ReduceMean**: Tests basic configurations
- **ReduceMax**: Tests basic configurations

## Adding New Tests

To add tests for a new operation:

1. Create a test class following the pattern in `test_reduction_ops.py`
2. Use `create_onnx_model()` from `test_utils.py` to create test models
3. Use `verify_tir_graph_structure()` to verify graph structure
4. Use `compare_tir_with_onnx()` to compare execution results
5. Add parametrized test cases for different configurations

Example:
```python
class TestNewOp:
    @pytest.mark.parametrize("input_shape", [(2, 3), (1, 4, 5)])
    def test_newop_basic(self, input_shape):
        onnx_model = create_onnx_model(...)
        transpiler = ONNXToForgeTranspiler(debug=True)
        tir_graph = transpiler.transpile(onnx_model)
        # ... verify and compare
```

