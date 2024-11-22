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
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 12, 100), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 32, 12, 100), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>broadcast</td>
      <td>Operand(type=Activation, name/shape=(1, 50, 1), dtype=float32)</td>
      <td>dim : -3<br>shape : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 12, 50), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 12, 50), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.concat validation] RuntimeError Tile padding along concatenated dim not supported for concat yet</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 12, 50), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 32, 12, 50), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.concat validation] RuntimeError Tile padding along concatenated dim not supported for concat yet</td>
    </tr>
    <tr>
      <td>cosine</td>
      <td>Operand(type=Activation, name/shape=(1, 12, 100), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 12, 100), dtype=float32)</td>
      <td>dim : -1<br>start : 50<br>stop : 100<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 12, 100), dtype=float32)</td>
      <td>dim : -1<br>start : 0<br>stop : 50<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 50, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_07, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 12, 100), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 1, 12, 100), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.tilize validation] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 12, 50), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_17, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sine</td>
      <td>Operand(type=Activation, name/shape=(1, 12, 100), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 50, 12), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=rotary_emb.inv_freq, dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(1, 50), dtype=float32)</td>
      <td>dim : 2</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.reshape mlir pipeline] RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(1, 12, 100), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
