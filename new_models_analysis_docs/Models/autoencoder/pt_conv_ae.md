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
      <td>Operand(type=Activation, name/shape=(1, 16, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(4, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 1, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(16, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.tilize_with_val_padding validation] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(4, 16, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.tilize_with_val_padding validation] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2dtranspose</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(4, 16, 2, 2), dtype=float32)</td>
      <td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>conv2dtranspose</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(16, 1, 2, 2), dtype=float32)</td>
      <td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>maxpool2d</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 28, 28), dtype=float32)</td>
      <td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.maxpool2d mlir pipeline] RuntimeError ttnn.max_pool2d currently only supports an input type of bfloat16 Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>maxpool2d</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 14, 14), dtype=float32)</td>
      <td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.maxpool2d mlir pipeline] RuntimeError ttnn.max_pool2d currently only supports an input type of bfloat16 Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 4, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=encoder_conv2d_1.bias, dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.reshape mlir pipeline] RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(16, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=encoder_conv2d_2.bias, dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.reshape mlir pipeline] RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(4, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=decoder_conv2d_2.bias, dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.reshape mlir pipeline] RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(1, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
