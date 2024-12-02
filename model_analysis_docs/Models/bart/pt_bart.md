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
      <td>abs</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_20, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(4096,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>cast</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=int64)</td>
      <td>dtype : torch.float32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>cast</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=uint1)</td>
      <td>dtype : torch.float32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)</td>
      <td>min : 0.0<br>max : 1.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>embedding</td>
      <td>Operand(type=Activation, name/shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(50265, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.embedding validation] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16 Input must be UINT32 or BFLOAT16</td>
    </tr>
    <tr>
      <td>embedding</td>
      <td>Operand(type=Activation, name/shape=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, name/shape=model.decoder.embed_positions.weight, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.embedding validation] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp weights.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>gelu</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 4096), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>greater</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_80, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn elementwise binary] RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(16, 256, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 4096), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1024,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][ttmetal vs Forge Output Data mismatch] AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(16, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16, 64, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.matmul] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp Mt % per_core_M == 0</td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16, 256, 64), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][ttmetal vs Forge Output Data mismatch] AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024, 4096), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.matmul] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op_multi_core_reuse_program_factory.cpp Mt % per_core_M == 0</td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(4096, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][ttmetal vs Forge Output Data mismatch] AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_10, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_90, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>repeatinterleave</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 1, 256), dtype=int64)</td>
      <td>repeats : 1<br>dim : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>repeatinterleave</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 1, 256), dtype=int64)</td>
      <td>repeats : 1<br>dim : 1</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>repeatinterleave</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 1, 256), dtype=int64)</td>
      <td>repeats : 256<br>dim : 2</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)</td>
      <td>shape : (256, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1024), dtype=float32)</td>
      <td>shape : (1, 256, 16, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(256, 1024), dtype=float32)</td>
      <td>shape : (1, 256, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 256, 64), dtype=float32)</td>
      <td>shape : (16, 256, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 256, 256), dtype=float32)</td>
      <td>shape : (1, 16, 256, 256)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 256, 256), dtype=float32)</td>
      <td>shape : (16, 256, 256)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 256, 64), dtype=float32)</td>
      <td>shape : (1, 16, 256, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 16, 64), dtype=float32)</td>
      <td>shape : (256, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 256), dtype=int64)</td>
      <td>shape : (1, 256)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>softmax</td>
      <td>Operand(type=Activation, name/shape=(16, 256, 256), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>subtract</td>
      <td>Operand(type=Constant, name/shape=const_60, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 1, 256, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=model.decoder.layers.0.self_attn.q_proj.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 16, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(16, 256, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(16, 64, 256), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 256, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=model.encoder.layers.0.fc1.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=model.encoder.layers.0.fc2.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(1, 256), dtype=int64)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 256), dtype=int64)</td>
      <td>dim : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
