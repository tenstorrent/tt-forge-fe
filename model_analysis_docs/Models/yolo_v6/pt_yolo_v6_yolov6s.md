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
      <td>Operand(type=Constant, name=const_0209, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 4, 4480), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 1120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 280), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 80, 4480), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 1120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 280), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 5880, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_208209, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 80), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 20), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 28, 40), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 448, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 224, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_12132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_15132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_18132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_21132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_24132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_27132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_33132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_36132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_39132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_42132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_45132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_48132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_51132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_54132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_57132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_61132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_62132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_63132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_65132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_66132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_68132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_69132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_71132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_72132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_74132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_78132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_81132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_84132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_87132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_90132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_93132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_94132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_95132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_96132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_97132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_99132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_102132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_105132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_106132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_107132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_108132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_109132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_111132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_114132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_115132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116132, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_117132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_118132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_120132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_121132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_122132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_123132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_124132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_126132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_127132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_129132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_130132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_134132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_135132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_136132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_137132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_138132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_139132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_140132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_141132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_142132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_143132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_144132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_145132, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2dTranspose</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56132, dtype=float32)</td>
      <td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>Conv2dTranspose</td>
      <td>Operand(type=Activation, shape=(1, 64, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_75132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76132, dtype=float32)</td>
      <td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 5880, 4), dtype=float32)</td>
      <td>dim : -1<br>start : 0<br>stop : 2<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 5880, 4), dtype=float32)</td>
      <td>dim : -1<br>start : 2<br>stop : 4<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
      <td>kernel_size : 5<br>stride : 1<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_206209, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 5880, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_207209, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 64, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 32, 224, 320), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 80, 56, 80), dtype=float32)</td>
      <td>shape : (1, 80, 4480)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 80, 28, 40), dtype=float32)</td>
      <td>shape : (1, 80, 1120)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 20), dtype=float32)</td>
      <td>shape : (1, 80, 280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 4, 56, 80), dtype=float32)</td>
      <td>shape : (1, 4, 4480)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 4, 28, 40), dtype=float32)</td>
      <td>shape : (1, 4, 1120)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 4, 14, 20), dtype=float32)</td>
      <td>shape : (1, 4, 280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 80, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 80, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Subtract</td>
      <td>Operand(type=Constant, name=const_0209, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Subtract</td>
      <td>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 4, 5880), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 80, 5880), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>