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
      <td>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 7, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(2, 8, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_10, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(2, 8, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2, 7, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(2048,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>broadcast</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 1, 7), dtype=int64)</td>
      <td>dim : -4<br>shape : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>broadcast</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 1, 7), dtype=int64)</td>
      <td>dim : -2<br>shape : 7</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>cast</td>
      <td>Operand(type=Activation, name/shape=(2, 7), dtype=int64)</td>
      <td>dtype : torch.int32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>cast</td>
      <td>Operand(type=Activation, name/shape=(1, 7), dtype=int64)</td>
      <td>dtype : torch.int32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>cast</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=int64)</td>
      <td>dtype : torch.float32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>cast</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=uint1)</td>
      <td>dtype : torch.float32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)</td>
      <td>min : 0.0<br>max : 1.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>embedding</td>
      <td>Operand(type=Activation, name/shape=(2, 7), dtype=int32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(49408, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.embedding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp weights.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>embedding</td>
      <td>Operand(type=Activation, name/shape=(1, 7), dtype=int32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(77, 512), dtype=float32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>greater</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_40, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn elementwise binary RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(16, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=clip_model.text_model.embeddings.position_ids, dtype=int64)</td>
      <td>dim : -1<br>start : 0<br>stop : 7<br>stride : 1</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: Runtime Data mismatch RuntimeError Tensor data type mismatch: expected got</td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(512,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: Runtime Shape mismatch RuntimeError Tensor shape mismatch: expected got</td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(14, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(16, 7, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16, 64, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(16, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16, 7, 64), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(14, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512, 2048), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(14, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2048, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_00, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_50, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_60, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2, 7, 2048), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2, 7), dtype=int64)</td>
      <td>shape : (2, 7)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 512), dtype=float32)</td>
      <td>shape : (14, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 512), dtype=float32)</td>
      <td>shape : (2, 7, 8, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.reshape validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(14, 512), dtype=float32)</td>
      <td>shape : (2, 7, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2, 8, 7, 64), dtype=float32)</td>
      <td>shape : (16, 7, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 7, 7), dtype=float32)</td>
      <td>shape : (2, 8, 7, 7)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2, 8, 7, 7), dtype=float32)</td>
      <td>shape : (2, 8, 7, 7)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2, 8, 7, 7), dtype=float32)</td>
      <td>shape : (16, 7, 7)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 7, 64), dtype=float32)</td>
      <td>shape : (2, 8, 7, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 8, 64), dtype=float32)</td>
      <td>shape : (14, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.reshape validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(14, 2048), dtype=float32)</td>
      <td>shape : (2, 7, 2048)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 2048), dtype=float32)</td>
      <td>shape : (14, 2048)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>sigmoid</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 2048), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>softmax</td>
      <td>Operand(type=Activation, name/shape=(16, 7, 7), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>subtract</td>
      <td>Operand(type=Constant, name/shape=const_20, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2, 1, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=clip_model.text_model.encoder.layers.0.self_attn.q_proj.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(2, 7, 8, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(16, 7, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(16, 64, 7), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(2, 8, 7, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=clip_model.text_model.encoder.layers.0.mlp.fc1.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=clip_model.text_model.encoder.layers.0.mlp.fc2.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(2, 7), dtype=int64)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(2, 1, 7), dtype=int64)</td>
      <td>dim : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
