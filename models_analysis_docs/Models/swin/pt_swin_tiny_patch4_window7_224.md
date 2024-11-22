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
      <td>Operand(type=Activation, name/shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(64, 49, 288), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(288,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 3, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(64, 49, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 3136, 96), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 3, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 64, 1, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(16, 49, 576), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 6, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(16, 49, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 784, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(768,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 6, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 16, 1, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(4, 49, 1152), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1152,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 12, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(4, 49, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 196, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1536,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 12, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 4, 1, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 2304), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(2304,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 24, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(768,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(3072,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 1000), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1000,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>advindex</td>
      <td>Operand(type=Parameter, name/shape=(169, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2401,), dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.embedding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16 Input must be UINT32 or BFLOAT16</td>
    </tr>
    <tr>
      <td>advindex</td>
      <td>Operand(type=Parameter, name/shape=(169, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2401,), dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.embedding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16 Input must be UINT32 or BFLOAT16</td>
    </tr>
    <tr>
      <td>advindex</td>
      <td>Operand(type=Parameter, name/shape=(169, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2401,), dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.embedding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16 Input must be UINT32 or BFLOAT16</td>
    </tr>
    <tr>
      <td>advindex</td>
      <td>Operand(type=Parameter, name/shape=(169, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(2401,), dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.embedding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp a.get_dtype() == DataType::UINT32 or a.get_dtype() == DataType::BFLOAT16 Input must be UINT32 or BFLOAT16</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 53, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 3, 56, 96), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 53, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 56, 3, 96), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.concat validation RuntimeError Tile padding along concatenated dim not supported for concat yet</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 3, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 53, 56, 96), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 3, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 56, 53, 96), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.concat validation RuntimeError Tile padding along concatenated dim not supported for concat yet</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 28, 28, 96), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 25, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 3, 28, 192), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 25, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 28, 3, 192), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.concat validation RuntimeError Tile padding along concatenated dim not supported for concat yet</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 3, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 25, 28, 192), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 3, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 28, 25, 192), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.concat validation RuntimeError Tile padding along concatenated dim not supported for concat yet</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 14, 14, 192), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 11, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 3, 14, 384), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 11, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 14, 3, 384), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.concat validation RuntimeError Tile padding along concatenated dim not supported for concat yet</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 3, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 11, 14, 384), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 3, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 14, 11, 384), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.concat validation RuntimeError Tile padding along concatenated dim not supported for concat yet</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 7, 7, 384), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96, 3, 4, 4), dtype=float32)</td>
      <td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>gelu</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 384), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>gelu</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 768), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>gelu</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 1536), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>gelu</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 3072), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 96), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(64, 49, 96), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(16, 49, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(4, 49, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 1536), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 3072), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 64, 3, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 64, 3, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 64, 3, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 3<br>stop : 56<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 3<br>stop : 56<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 53<br>stop : 56<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 53<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 53<br>stop : 56<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 53<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 56<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 1<br>stop : 56<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 56<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 1<br>stop : 56<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 16, 6, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 16, 6, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 16, 6, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 3<br>stop : 28<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 3<br>stop : 28<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 25<br>stop : 28<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 25<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 25<br>stop : 28<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 25<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 28<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 1<br>stop : 28<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 28<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 1<br>stop : 28<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 4, 12, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 4, 12, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 4, 12, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 3<br>stop : 14<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 3<br>stop : 14<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 11<br>stop : 14<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 11<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 11<br>stop : 14<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 11<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 14<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 1<br>stop : 14<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 7, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 14<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(1, 7, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 1<br>stop : 14<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 1, 24, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 1, 24, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>index</td>
      <td>Operand(type=Activation, name/shape=(3, 1, 24, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(192,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(768,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1536,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>layernorm</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(768,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 288), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(192, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192, 32, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(192, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192, 49, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 96), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384, 96), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192, 576), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(96, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 32, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(96, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 49, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(768, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(768, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384, 1152), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(48, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(48, 32, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(48, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(48, 49, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384, 1536), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1536, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1536, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(768, 2304), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(24, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(24, 32, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(24, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(24, 49, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(768, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(768, 3072), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(3072, 768), dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(768, 1000), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_00, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_20, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_40, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name/shape=const_100, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reduceavg</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)</td>
      <td>dim : -2<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 56, 56), dtype=float32)</td>
      <td>shape : (1, 96, 3136, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 96), dtype=float32)</td>
      <td>shape : (1, 8, 7, 8, 7, 96)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 3136, 96), dtype=float32)</td>
      <td>shape : (1, 56, 56, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 8, 8, 7, 7, 96), dtype=float32)</td>
      <td>shape : (3136, 96)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(3136, 288), dtype=float32)</td>
      <td>shape : (64, 49, 288)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(64, 49, 288), dtype=float32)</td>
      <td>shape : (64, 49, 3, 3, 32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 32), dtype=float32)</td>
      <td>shape : (192, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(192, 49, 49), dtype=float32)</td>
      <td>shape : (64, 3, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=layers.0.blocks.0.attn.relative_position_index, dtype=int64)</td>
      <td>shape : (2401,)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2401, 3), dtype=float32)</td>
      <td>shape : (49, 49, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 49), dtype=float32)</td>
      <td>shape : (192, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 49), dtype=float32)</td>
      <td>shape : (1, 64, 3, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 32, 49), dtype=float32)</td>
      <td>shape : (192, 32, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(192, 49, 32), dtype=float32)</td>
      <td>shape : (64, 3, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(64, 49, 3, 32), dtype=float32)</td>
      <td>shape : (3136, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.reshape validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(3136, 96), dtype=float32)</td>
      <td>shape : (64, 49, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(64, 49, 96), dtype=float32)</td>
      <td>shape : (1, 8, 8, 7, 7, 96)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 8, 7, 8, 7, 96), dtype=float32)</td>
      <td>shape : (1, 3136, 96)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 8, 7, 8, 7, 96), dtype=float32)</td>
      <td>shape : (1, 56, 56, 96)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>shape : (1, 8, 7, 8, 7, 96)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>shape : (1, 3136, 96)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 3, 49, 49), dtype=float32)</td>
      <td>shape : (64, 3, 49, 49)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 384), dtype=float32)</td>
      <td>shape : (1, 784, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 192), dtype=float32)</td>
      <td>shape : (1, 4, 7, 4, 7, 192)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 784, 192), dtype=float32)</td>
      <td>shape : (1, 28, 28, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 4, 7, 7, 192), dtype=float32)</td>
      <td>shape : (784, 192)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(784, 576), dtype=float32)</td>
      <td>shape : (16, 49, 576)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 49, 576), dtype=float32)</td>
      <td>shape : (16, 49, 3, 6, 32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 32), dtype=float32)</td>
      <td>shape : (96, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(96, 49, 49), dtype=float32)</td>
      <td>shape : (16, 6, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2401, 6), dtype=float32)</td>
      <td>shape : (49, 49, 6)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 49), dtype=float32)</td>
      <td>shape : (96, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 49), dtype=float32)</td>
      <td>shape : (1, 16, 6, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 32, 49), dtype=float32)</td>
      <td>shape : (96, 32, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(96, 49, 32), dtype=float32)</td>
      <td>shape : (16, 6, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 49, 6, 32), dtype=float32)</td>
      <td>shape : (784, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.reshape validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(784, 192), dtype=float32)</td>
      <td>shape : (16, 49, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(16, 49, 192), dtype=float32)</td>
      <td>shape : (1, 4, 4, 7, 7, 192)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 7, 4, 7, 192), dtype=float32)</td>
      <td>shape : (1, 784, 192)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 7, 4, 7, 192), dtype=float32)</td>
      <td>shape : (1, 28, 28, 192)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>shape : (1, 4, 7, 4, 7, 192)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>shape : (1, 784, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 6, 49, 49), dtype=float32)</td>
      <td>shape : (16, 6, 49, 49)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 768), dtype=float32)</td>
      <td>shape : (1, 196, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 384), dtype=float32)</td>
      <td>shape : (1, 2, 7, 2, 7, 384)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 196, 384), dtype=float32)</td>
      <td>shape : (1, 14, 14, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 2, 2, 7, 7, 384), dtype=float32)</td>
      <td>shape : (196, 384)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(196, 1152), dtype=float32)</td>
      <td>shape : (4, 49, 1152)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(4, 49, 1152), dtype=float32)</td>
      <td>shape : (4, 49, 3, 12, 32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 32), dtype=float32)</td>
      <td>shape : (48, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(48, 49, 49), dtype=float32)</td>
      <td>shape : (4, 12, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2401, 12), dtype=float32)</td>
      <td>shape : (49, 49, 12)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 49), dtype=float32)</td>
      <td>shape : (48, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 49), dtype=float32)</td>
      <td>shape : (1, 4, 12, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 32, 49), dtype=float32)</td>
      <td>shape : (48, 32, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(48, 49, 32), dtype=float32)</td>
      <td>shape : (4, 12, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(4, 49, 12, 32), dtype=float32)</td>
      <td>shape : (196, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.reshape validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(196, 384), dtype=float32)</td>
      <td>shape : (4, 49, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(4, 49, 384), dtype=float32)</td>
      <td>shape : (1, 2, 2, 7, 7, 384)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 2, 7, 2, 7, 384), dtype=float32)</td>
      <td>shape : (1, 196, 384)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 2, 7, 2, 7, 384), dtype=float32)</td>
      <td>shape : (1, 14, 14, 384)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>shape : (1, 2, 7, 2, 7, 384)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: mlir generation failure RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>shape : (1, 196, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 12, 49, 49), dtype=float32)</td>
      <td>shape : (4, 12, 49, 49)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 7, 7, 1536), dtype=float32)</td>
      <td>shape : (1, 49, 1536)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 2304), dtype=float32)</td>
      <td>shape : (1, 49, 3, 24, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.reshape validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 49, 32), dtype=float32)</td>
      <td>shape : (24, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(24, 49, 49), dtype=float32)</td>
      <td>shape : (1, 24, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(2401, 24), dtype=float32)</td>
      <td>shape : (49, 49, 24)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 49, 49), dtype=float32)</td>
      <td>shape : (24, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 32, 49), dtype=float32)</td>
      <td>shape : (24, 32, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(24, 49, 32), dtype=float32)</td>
      <td>shape : (1, 24, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 24, 32), dtype=float32)</td>
      <td>shape : (49, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.reshape validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(49, 768), dtype=float32)</td>
      <td>shape : (1, 49, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 768), dtype=float32)</td>
      <td>shape : (1, 49, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>softmax</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 49), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>softmax</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 49), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>softmax</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 49), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>softmax</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 49, 49), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>squeeze</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 3136, 1), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>squeeze</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 3, 49, 32), dtype=float32)</td>
      <td>dim : -5</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>squeeze</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 6, 49, 32), dtype=float32)</td>
      <td>dim : -5</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>squeeze</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 12, 49, 32), dtype=float32)</td>
      <td>dim : -5</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>squeeze</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 24, 49, 32), dtype=float32)</td>
      <td>dim : -5</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>squeeze</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 768), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 3136), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 8, 7, 8, 7, 96), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.0.blocks.0.attn.qkv.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(64, 49, 3, 3, 32), dtype=float32)</td>
      <td>dim0 : -5<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 49, 64, 3, 32), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 64, 49, 3, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(192, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(49, 49, 3), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 49, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(64, 3, 49, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(192, 32, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.0.blocks.0.attn.proj.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 8, 8, 7, 7, 96), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.0.blocks.0.mlp.fc1.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.0.blocks.0.mlp.fc2.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.0.downsample.reduction.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 7, 4, 7, 192), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.1.blocks.0.attn.qkv.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(16, 49, 3, 6, 32), dtype=float32)</td>
      <td>dim0 : -5<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 49, 16, 6, 32), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 16, 49, 6, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(96, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(49, 49, 6), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(6, 49, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(16, 6, 49, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(96, 32, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.1.blocks.0.attn.proj.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 4, 7, 7, 192), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.1.blocks.0.mlp.fc1.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.1.blocks.0.mlp.fc2.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.1.downsample.reduction.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 2, 7, 2, 7, 384), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.2.blocks.0.attn.qkv.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(4, 49, 3, 12, 32), dtype=float32)</td>
      <td>dim0 : -5<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 49, 4, 12, 32), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 4, 49, 12, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(48, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(49, 49, 12), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(12, 49, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(4, 12, 49, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(48, 32, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.2.blocks.0.attn.proj.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 2, 2, 7, 7, 384), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.2.blocks.0.mlp.fc1.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.2.blocks.0.mlp.fc2.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.2.downsample.reduction.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.3.blocks.0.attn.qkv.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 49, 3, 24, 32), dtype=float32)</td>
      <td>dim0 : -5<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 49, 1, 24, 32), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(3, 1, 49, 24, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(24, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(49, 49, 24), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(24, 49, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 49, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=(24, 32, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.3.blocks.0.attn.proj.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.3.blocks.0.mlp.fc1.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=layers.3.blocks.0.mlp.fc2.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=head.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=patch_embed.proj.bias, dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(96, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(3, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=layers.0.blocks.1.attn_mask, dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(64, 1, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(6, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=layers.1.blocks.1.attn_mask, dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(16, 1, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(12, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=layers.2.blocks.1.attn_mask, dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(4, 1, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(24, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
