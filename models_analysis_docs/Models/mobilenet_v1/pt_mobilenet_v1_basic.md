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
      <td>Operand(type=Activation, name/shape=model.0.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_1214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=model.1.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_7214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=model.2.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_13214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=model.4.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_25214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(256,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=model.6.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_37214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 2, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=model.12.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_73214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 1024, 2, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(9,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>avgpool2d</td>
      <td>Operand(type=Activation, name/shape=(1, 1024, 2, 2), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 3, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(32, 3, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(64, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(128, 64, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(128, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(128, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(128, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(256, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(256, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(512, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 512<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 512<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(512, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 2, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1024, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 1024, 2, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1024<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 1024, 2, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1024, 1024, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>matmul</td>
      <td>Operand(type=Activation, name/shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024, 9), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_0214, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=model.0.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_2214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_6214, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=model.1.model.3.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_8214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_12214, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=model.2.model.3.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_14214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_24214, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(256,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=model.4.model.3.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_26214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(256,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_36214, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=model.6.model.3.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_38214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 2, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_72214, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 1024, 2, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=model.12.model.3.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_74214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 32, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 32, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 16, 16), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 16, 16), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 128, 8, 8), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 8, 8), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 4, 4), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 4, 4), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 2, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 1024, 2, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=model.1.model.0.weight, dtype=float32)</td>
      <td>shape : (32, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=model.2.model.0.weight, dtype=float32)</td>
      <td>shape : (64, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=model.3.model.0.weight, dtype=float32)</td>
      <td>shape : (128, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=model.5.model.0.weight, dtype=float32)</td>
      <td>shape : (256, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=model.7.model.0.weight, dtype=float32)</td>
      <td>shape : (512, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=model.13.model.0.weight, dtype=float32)</td>
      <td>shape : (1024, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=(1, 1024, 1, 1), dtype=float32)</td>
      <td>shape : (1, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>transpose</td>
      <td>Operand(type=Activation, name/shape=fc.weight, dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(32,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(32, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(64,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(64, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(128,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(128, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(256,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(256, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(512,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(512, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(1024,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(1024, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
