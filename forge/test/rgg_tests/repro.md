
### Test case 1 - Unexpected high pcc

```bash
pytest forge/test/rgg_tests/test_0001_forge_unexpected_high_pcc.py
```

Outcome: PASSED

Expected: yes


---


### Test case 2 - Expected low pcc

```bash
pytest forge/test/rgg_tests/test_0002_forge_expected_low_pcc.py
```
Outcome: FAILED

Expected: yes

ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden)


---


### Test case 3 - Different input types

```bash
pytest forge/test/rgg_tests/test_0003_forge_concat_different_input_types.py
```

Outcome: FAILED

Expected: yes

RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp:42: in_ref.dtype() == first_input.dtype()

Info: All Tensors should have same dtypes.


---
