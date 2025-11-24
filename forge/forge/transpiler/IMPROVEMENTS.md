# ONNX Transpiler Improvements Based on TVM Frontend Analysis

This document summarizes improvements to the ONNX to Forge transpiler based on analysis of the TVM Relay ONNX frontend implementation.

## Table of Contents

1. [Overview](#overview)
2. [Critical Improvements](#critical-improvements)
3. [High Priority Improvements](#high-priority-improvements)
4. [Medium Priority Improvements](#medium-priority-improvements)
5. [Low Priority Improvements](#low-priority-improvements)
6. [Implementation Notes](#implementation-notes)

## Overview

The TVM ONNX frontend (`third_party/tvm/python/tvm/relay/frontend/onnx.py`) provides a mature, production-ready implementation with ~7600 lines of code. This analysis identifies patterns and best practices that can improve our transpiler.

**Key Differences:**
- **TVM**: Uses Relay expressions, supports multiple outputs via `TupleWrapper`
- **Our Transpiler**: Uses TIR nodes, **must enforce single output per node** (Forge constraint)
- **Adaptation Strategy**: Split multi-output ONNX operations into multiple TIR nodes

## Critical Improvements

### 1. Single Output Enforcement in TIRNode

**Current State:**
- `TIRNode.__init__` accepts `outputs: List[str]` but only uses `outputs[0]`
- Inconsistent with Forge's single-output constraint

**Required Changes:**
- Change `TIRNode.__init__`: `outputs: List[str]` → `output: str`
- Update all operation nodes to use single output
- Update `emit()`: Use `self.output` instead of `self.outputs[0]`
- Update all converter methods to return single-output nodes

**Files to Modify:**
- `forge/forge/transpiler/ir/nodes.py`
- All files in `forge/forge/transpiler/ir/operations/`
- `forge/forge/transpiler/frontends/onnx/engine.py`

---

### 2. Multiple Output Handling (Forge Constraint)

**Problem:**
- ONNX operations like `TopK`, `Split`, `Unique` return multiple outputs
- Forge operations only support single output
- Need to split multi-output operations into multiple TIR nodes

**TVM Approach:**
- Uses `TupleWrapper` to handle multiple outputs
- Filters unused outputs before storing
- Stores each output separately: `self._nodes[k] = op[i]`

**Our Approach:**
```python
def _convert_topk(self, node_proto, input_tensors, output_tensors, attrs, node_index):
    """Convert ONNX TopK to multiple TIR nodes (one per output)."""
    used_outputs = self._get_used_outputs(node_proto)  # [True, True] or [True, False]
    nodes = []
    
    if used_outputs[0]:  # values output used
        nodes.append(TopKValuesNode.create(
            name=f"{node_proto.name}_values",
            output=node_proto.output[0],
            inputs=node_proto.input,
            ...
        ))
    if used_outputs[1]:  # indices output used
        nodes.append(TopKIndicesNode.create(
            name=f"{node_proto.name}_indices", 
            output=node_proto.output[1],
            inputs=node_proto.input,
            ...
        ))
    return nodes
```

**Required Implementation:**
1. Add `_get_used_outputs()` method to check which outputs are consumed
2. Build consumer map: Track which tensors are used by other nodes
3. Check graph outputs: Mark tensors that are final outputs
4. Filter before conversion: Only create nodes for used outputs
5. Split operations: Create separate TIR nodes for each needed output

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`
- Add new operation nodes for split outputs (e.g., `TopKValuesNode`, `TopKIndicesNode`)

---

### 3. Parameters vs Constants Distinction

**Current State:**
- Uses `initializers` dict (ONNX-specific terminology)
- No distinction between trainable parameters and constants

**TVM Approach:**
- Separates `_params` (trainable) and constants (non-trainable)
- Heuristics in `_parse_graph_initializers()`:
  ```python
  if "constant" in name.lower() or (
      "weight" not in name and "bias" not in name
      and ("int" in dtype or "bool" in dtype)
  ) or len(shape) == 0:
      # Constant
  else:
      # Parameter (trainable)
  ```
- `freeze_params` flag: If True, all become constants

**Required Changes:**
1. Update `TIRGraph`: Replace `self.initializers` with `self.params` and `self.constants`
2. Implement heuristics to distinguish parameters from constants
3. Add `freeze_params` option to `ONNXToForgeTranspiler.__init__()`
4. Update converter methods to access `params` and `constants` separately

**Files to Modify:**
- `forge/forge/transpiler/core/graph.py`
- `forge/forge/transpiler/frontends/onnx/engine.py`

**Example:**
```python
# In TIRGraph
self.params: Dict[str, torch.Tensor] = {}      # Trainable weights
self.constants: Dict[str, torch.Tensor] = {}   # Non-trainable values

# In transpile()
for initializer in graph_proto.initializer:
    tensor = convert_to_torch_tensor(initializer)
    if is_constant(initializer.name, tensor):
        tir_graph.constants[initializer.name] = tensor
    else:
        tir_graph.params[initializer.name] = tensor
```

---

### 4. Opset Version Support

**Current State:**
- No opset version handling
- All converters assume latest ONNX opset behavior

**TVM Approach:**
- Base class: `OnnxOpConverter` with `get_converter(opset)` classmethod
- Versioned implementations: `_impl_v1`, `_impl_v11`, `_impl_v13`, etc.
- Dynamic selection: Finds highest version ≤ opset
- Convert map: `_get_convert_map(opset)` builds map per opset

**Required Implementation:**
1. Create `OnnxOpConverter` base class
2. Add versioned converter methods: `_convert_conv_v1`, `_convert_conv_v11`, etc.
3. Extract opset from model: `model.opset_import[0].version` (default to 1)
4. Build dynamic convert map: `_get_convert_map(opset)`
5. Update converter methods to use versioned implementations

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`
- Create new file: `forge/forge/transpiler/frontends/onnx/converters/base.py`

**Example:**
```python
class OnnxOpConverter:
    @classmethod
    def get_converter(cls, opset):
        """Get converter for given opset version."""
        versions = [int(m.replace("_impl_v", "")) 
                   for m in dir(cls) if m.startswith("_impl_v")]
        versions = sorted(versions + [opset])
        version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]
        return getattr(cls, f"_impl_v{version}")

class ConvConverter(OnnxOpConverter):
    @classmethod
    def _impl_v1(cls, ...):  # Opset 1-10
        ...
    
    @classmethod
    def _impl_v11(cls, ...):  # Opset 11+
        ...
```

**Operations Needing Version Support:**
- `Pad`: v1 (pads as attr), v2 (pads as attr), v11 (pads as input)
- `Split`: v1 (split as attr), v13 (split as input)
- `Squeeze`: v1 (axes as attr), v13 (axes as input)
- `Unsqueeze`: Similar version differences

---

## High Priority Improvements

### 5. Early Validation

**TVM Approach:**
- `_check_for_unsupported_ops()`: Validates all ops before processing
- Raises clear errors: `tvm.error.OpNotImplemented` with list of unsupported ops
- Checks against convert map before node construction

**Required Implementation:**
```python
def _check_for_unsupported_ops(self, graph):
    """Check for unsupported operations before processing."""
    convert_map = self._get_convert_map(self.opset)
    unsupported_ops = set()
    for node in graph.node:
        op_name = node.op_type
        if op_name not in convert_map and op_name != "Constant":
            unsupported_ops.add(op_name)
    if unsupported_ops:
        msg = f"The following operators are not supported: {', '.join(unsupported_ops)}"
        raise NotImplementedError(msg)
```

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`

---

### 6. Output Usage Analysis

**TVM Approach:**
- Uses `analysis.free_vars(outputs)` to find which params are actually used
- Only includes needed params in final function
- Filters unused outputs before storing

**Required Implementation:**
1. Build consumer map: For each tensor, track which nodes consume it
2. Check graph outputs: Mark tensors that are final outputs
3. Utility method: `_get_used_outputs(node_proto)` returns list of used output indices
4. Apply before conversion: Filter `node.output` to only used outputs

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`

**Example:**
```python
def _build_consumer_map(self, graph):
    """Build map of tensor -> list of consuming nodes."""
    consumer_map = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in consumer_map:
                consumer_map[input_name] = []
            consumer_map[input_name].append(node.name)
    return consumer_map

def _get_used_outputs(self, node_proto):
    """Determine which outputs of a node are actually used."""
    used = []
    for i, output_name in enumerate(node_proto.output):
        if output_name == "":
            used.append(False)
        elif output_name in self.consumer_map or output_name in self.graph_outputs:
            used.append(True)
        else:
            used.append(False)
    return used
```

---

### 7. Input Validation and Graceful Handling

**TVM Approach:**
- `onnx_input` class: Returns `None` for out-of-bounds indices
- Input validation: Checks if input exists in `_nodes` before using
- Renaming: `_renames` dict for handling renamed tensors

**Required Implementation:**
1. Add input validation in converter methods
2. Handle missing inputs gracefully (return None or raise clear error)
3. Add renaming support if needed

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`

---

### 8. Constant Node Handling

**TVM Approach:**
- `Constant` op: Converts ONNX Constant nodes to Relay constants
- Handled in `_construct_nodes`: Special case for Constant nodes
- Stores constant value directly in `_nodes`

**Required Implementation:**
```python
def _convert_constant(self, node_proto, ...):
    """Handle ONNX Constant nodes."""
    # Extract value from attributes
    value_attr = [a for a in node_proto.attribute if a.name == "value"][0]
    np_array = numpy_helper.to_array(value_attr.t)
    torch_tensor = torch.from_numpy(np_array)
    
    # Store as constant in graph
    output_name = node_proto.output[0]
    self.tir_graph.constants[output_name] = torch_tensor
    
    # Return Identity node or skip node creation
    return []  # No TIR node needed, value is in constants
```

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`

---

### 9. Identity Node Handling

**TVM Approach:**
- `Identity` op: Maps to `Renamer("copy")` - simple pass-through
- In `_identity_list`: Operators that don't need conversion

**Required Implementation:**
- Handle Identity as pass-through (skip node creation or create Identity node)
- Map input directly to output in graph

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`

---

## Medium Priority Improvements

### 10. Enhanced Attribute Parsing

**TVM Approach:**
- Comprehensive `_parse_attr()`: Handles all ONNX attribute types
- Error handling: Raises `ValueError` if attribute can't be parsed
- Handles: f, i, s, floats, ints, strings, t, tensors, graphs

**Current State:**
- Basic attribute extraction in `extract_attributes()`
- May miss edge cases

**Improvement:**
- Enhance `extract_attributes()` with better error handling
- Add support for all ONNX attribute types
- Add validation for attribute values

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/converters/attributes.py`

---

### 11. Shape Inference Utilities

**TVM Approach:**
- `infer_shape()`, `infer_type()`, `infer_value()`: Comprehensive inference
- `fold_constant()`: Constant folding after conversion
- Dynamic shape warnings: Warns about unknown dimensions

**Required Implementation:**
- Add shape inference utilities
- Add constant folding pass
- Add warnings for dynamic/unknown shapes

**Files to Create:**
- `forge/forge/transpiler/frontends/onnx/utils/shape_inference.py`

---

### 12. Constant Folding

**TVM Approach:**
- `fold_constant()`: Folds constants after conversion
- Applied to both single and multi-output ops

**Required Implementation:**
- Add constant folding pass after node creation
- Fold operations that can be computed at compile time

**Files to Create:**
- `forge/forge/transpiler/frontends/onnx/passes/constant_folding.py`

---

### 13. Better Node Naming

**TVM Approach:**
- `get_source_name()`: Generates names for unnamed nodes
- `_parse_value_proto()`: Handles both ValueProto and raw strings
- `_op_type_dict`: Tracks node types for naming

**Required Implementation:**
- Improve node naming for unnamed ONNX nodes
- Use op_type + index pattern: `{op_type}_{index}`

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`

---

### 14. Free Variable Analysis

**TVM Approach:**
- Uses `analysis.free_vars(outputs)` to find which params are actually used
- Only includes needed params in final function

**Required Implementation:**
- Implement free variable analysis for TIR graphs
- Filter unused params/constants from final graph

**Files to Create:**
- `forge/forge/transpiler/core/analysis.py`

---

### 15. Configuration System

**TVM Approach:**
- `ONNX_DEFAULT_CONFIGS`: Global config dict
- `convert_config` parameter: Allows overriding defaults
- Example: `use_nt_batch_matmul` flag

**Required Implementation:**
- Add configuration system for conversion options
- Allow user to override defaults

**Files to Create:**
- `forge/forge/transpiler/frontends/onnx/config.py`

---

## Low Priority Improvements

### 16. GraphProto Class Pattern (Architectural)

**TVM Approach:**
- `GraphProto` class maintains state (`_nodes`, `_params`, `_inputs`, `_renames`)
- Context manager pattern: `with g:` for scoped access

**Consideration:**
- Current procedural approach works, but class-based might be cleaner
- Low priority - architectural refactoring

---

### 17. Model Validation (Optional)

**TVM Approach:**
- ONNX model checker: `onnx.checker.check_model()` before conversion
- Catches invalid models early
- Warnings instead of errors (graceful degradation)

**Required Implementation:**
- Add optional ONNX model validation
- Make it optional (some models may have minor issues but still work)

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`

---

### 18. Custom Metadata Dict

**TVM Approach:**
- `attr["tvm_custom"]`: Stores metadata (name, num_outputs) for converters
- Passed to converter functions for context

**Consideration:**
- May be useful for debugging and converter context
- Low priority

---

### 19. Per-Input Dtype Support

**TVM Approach:**
- Supports dict of dtypes per input
- `dtype` parameter can be str or dict

**Required Implementation:**
- Allow per-input dtype specification
- Useful for models with mixed input types

**Files to Modify:**
- `forge/forge/transpiler/frontends/onnx/engine.py`

---

### 20. Export Renamed Model (Debugging)

**TVM Approach:**
- `export_node_renamed_model_path`: Exports model with renamed nodes
- Useful for debugging and span mapping

**Consideration:**
- Useful for debugging
- Low priority

---

## Implementation Notes

### Key Architectural Differences

1. **Output Handling:**
   - **TVM**: Uses `TupleWrapper` for multiple outputs
   - **Our**: Must split into multiple TIR nodes (Forge constraint)

2. **Node Representation:**
   - **TVM**: Relay expressions (functional)
   - **Our**: TIR nodes (object-oriented with static factory methods)

3. **Parameter Storage:**
   - **TVM**: `_params` dict + constants as expressions
   - **Our**: Should use `params` and `constants` dicts in `TIRGraph`

### Implementation Priority

1. **Critical (Must Have):**
   - Single output enforcement
   - Multiple output handling
   - Parameters vs constants
   - Opset version support

2. **High Priority:**
   - Early validation
   - Output usage analysis
   - Input validation
   - Constant/Identity node handling

3. **Medium Priority:**
   - Enhanced attribute parsing
   - Shape inference utilities
   - Constant folding
   - Better node naming
   - Free variable analysis
   - Configuration system

4. **Low Priority:**
   - Architectural refactoring
   - Optional features
   - Debugging utilities

### Testing Strategy

For each improvement:
1. Add unit tests for new functionality
2. Test with various ONNX models (different opsets)
3. Verify backward compatibility
4. Test edge cases (empty outputs, unused outputs, etc.)

---

## References

- TVM ONNX Frontend: `third_party/tvm/python/tvm/relay/frontend/onnx.py`
- ONNX Specification: https://github.com/onnx/onnx/blob/main/docs/IR.md
- ONNX Opset Versions: https://github.com/onnx/onnx/blob/main/docs/Versioning.md

---

## Implementation Checklist

### Critical
- [ ] Enforce single output in TIRNode
- [ ] Implement multiple output handling (filter + split)
- [ ] Replace initializers with params/constants
- [ ] Add opset version support

### High Priority
- [ ] Add early validation
- [ ] Implement output usage analysis
- [ ] Add input validation
- [ ] Handle Constant nodes
- [ ] Handle Identity nodes

### Medium Priority
- [ ] Enhance attribute parsing
- [ ] Add shape inference utilities
- [ ] Implement constant folding
- [ ] Improve node naming
- [ ] Add free variable analysis
- [ ] Add configuration system

### Low Priority
- [ ] Consider GraphProto class pattern
- [ ] Add optional model validation
- [ ] Add custom metadata dict
- [ ] Support per-input dtype
- [ ] Add export renamed model

