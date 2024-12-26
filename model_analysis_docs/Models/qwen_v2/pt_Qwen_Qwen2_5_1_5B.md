<h1>Unique ops configuration and compiler support info</h1>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th colspan="3" halign="left">Operation Details</th>
      <th colspan="4" halign="left">Component Passing Check</th>
      <th>Issues</th>
    </tr>
    <tr>
      <th>Name</th>
      <th>Operands</th>
      <th>Arguments</th>
      <th>Forge-Fe</th>
      <th>MLIR</th>
      <th>Metalium</th>
      <th>N/A</th>
      <th>Failure Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 29, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 29, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 29, 64), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 29, 64), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Cosine</td>
      <td>Operand(type=Activation, shape=(1, 29, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Embedding</td>
      <td>Operand(type=Activation, shape=(1, 29), dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(151936, 1536), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.embedding validation] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16 Input must be UINT32 or BFLOAT16</td>
    </tr>
    <tr>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 29), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
      <td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
      <td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)</td>
      <td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)</td>
      <td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1536), dtype=float32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(12, 29, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 128, 29), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(12, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 29, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 8960), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 29, 8960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8960, 1536), dtype=float32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 151936), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 29, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Parameter, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 29, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 29, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 29, 8960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 29, 8960), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1, 29, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)</td>
      <td>dim : -1<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)</td>
      <td>repeats : 1<br>dim : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)</td>
      <td>repeats : 1<br>dim : 2</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 2, 1, 29, 128), dtype=float32)</td>
      <td>repeats : 1<br>dim : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 2, 1, 29, 128), dtype=float32)</td>
      <td>repeats : 6<br>dim : 2</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)</td>
      <td>shape : (29, 1536)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)</td>
      <td>shape : (1, 29, 12, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(29, 1536), dtype=float32)</td>
      <td>shape : (1, 29, 1536)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
      <td>shape : (12, 29, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(29, 256), dtype=float32)</td>
      <td>shape : (1, 29, 256)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 29, 256), dtype=float32)</td>
      <td>shape : (1, 29, 2, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 2, 6, 29, 128), dtype=float32)</td>
      <td>shape : (12, 29, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 2, 6, 29, 128), dtype=float32)</td>
      <td>shape : (1, 12, 29, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(12, 29, 29), dtype=float32)</td>
      <td>shape : (1, 12, 29, 29)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 29), dtype=float32)</td>
      <td>shape : (12, 29, 29)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 29), dtype=float32)</td>
      <td>shape : (12, 128, 29)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(12, 29, 128), dtype=float32)</td>
      <td>shape : (1, 12, 29, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 29, 12, 128), dtype=float32)</td>
      <td>shape : (29, 1536)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(29, 8960), dtype=float32)</td>
      <td>shape : (1, 29, 8960)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 29, 8960), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sine</td>
      <td>Operand(type=Activation, shape=(1, 29, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 29), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1, 29, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1536, 1536), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(256, 1536), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(8960, 1536), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1536, 8960), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(151936, 1536), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 29, 12, 128), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 64, 29), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 29, 2, 128), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(12, 29, 128), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(12, 128, 29), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Constant, name=model.rotary_emb.inv_freq, dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1, 64), dtype=float32)</td>
      <td>dim : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1, 29, 128), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)</td>
      <td>dim : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
