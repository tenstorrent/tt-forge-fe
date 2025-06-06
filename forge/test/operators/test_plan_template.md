GENERAL OP SUPPORT TEST PLAN:
1. Operand type - any supported type (e.g. add, matmul, conv2d, etc.)
2. Operand source(s):
   - (-)  2.1 From another op
      - Operator -> input
   - (-)  2.2 From DRAM queue
      - Operator is first node in network
      - Input_queue flag = false
   - (-)  2.3 Const Inputs (const eval pass)
      - Operator where all inputs are constants.
   - (-)  2.4 From host
      - Input tensor as input of network
      - Operator is first node in network
      - Input_queue flag = true
3. Tensor ranks:
   - (-)  3.1 Full tensor (i.e. full expected shape)
      - 3-4 by default P1 (high prioriy)
      - 2, 5, ++ include P2 (lower prioriy)
   - (-)  3.2 Tensor reduce on one or more dims to 1
      - Vector
      - Only one dim is not equal to 1
   - (-)  3.3 Scalar P2
      - Create tensor of dimension equal to 0 (tensor from scalar) or just to use scalar as simple value
4. Operand / output size of dimensions (few examples of each, 10 values total)
   - (-)  4.1 Divisible by 32
   - (-)  4.2 Prime numbers
   - (-)  4.3 Very large (thousands, 10s of thousands)
      - 100x100, 100x1000
      - maybe nightly only
   - (-)  4.4 Extreme ratios between height/width
   - (-)  4.5 ...probably many more interesting combinations here
5. Data format - all supported formats
   - (-)  5.1 Output DF
   - (-)  5.2 Intermediate DF
   - (-)  5.3 Accumulation DF
   - (-)  5.4 Operand DFs
      - Fix HiFi4 for math fidelity value
   - (-) 6. Math fidelity - LoFi, HiFi2a, Hifi2b, Hifi3, Hifi4
      - Fix fp16b (default) for data format value
   - (-) 7. Special attributes - if applicable.. like approx_mode for Exp, for example
   - (-) 8. Special cases - if applicable
9. Variable number of operands - if applicable
   - (-) Few representative values
   - (-) Reuse inputs for selected operators
