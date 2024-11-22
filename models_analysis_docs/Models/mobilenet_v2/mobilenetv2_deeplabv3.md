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
      <td>Operand(type=Activation, name/shape=mobilenet_v2.conv_stem.first_conv.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_1430, dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.conv_stem.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_4430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.conv_stem.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_7430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.0.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_10430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.0.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_13430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
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
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.0.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_16430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(24,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(24,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(24, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.1.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_19430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(144, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.1.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_22430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.1.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_25430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 24, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.2.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_28430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.2.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_31430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(144, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.2.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_34430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.3.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_37430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.3.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_40430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.3.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_43430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 32, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.4.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_46430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.4.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_49430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.4.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_52430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.5.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_55430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.5.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_58430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.5.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_61430, dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.6.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_64430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.6.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_67430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.6.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_70430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 64, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.7.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_73430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.7.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_76430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.7.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_79430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.8.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_82430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.8.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_85430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.8.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_88430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.9.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_91430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.9.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_94430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.9.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_97430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.10.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_100430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(576,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(576, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.10.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_103430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.10.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_106430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 96, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.11.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_109430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.11.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_112430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.11.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_115430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.12.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_118430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.12.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_121430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.12.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_124430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.13.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_127430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(960,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.13.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_130430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.13.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_133430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 160, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.14.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_136430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.14.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_139430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.14.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_142430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.15.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_145430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.15.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_148430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.15.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_151430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(320, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=segmentation_head.conv_pool.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_154430, dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=segmentation_head.conv_aspp.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_157430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=segmentation_head.conv_projection.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_160430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>add</td>
      <td>Operand(type=Activation, name/shape=(1, 21, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(21, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>avgpool2d</td>
      <td>Operand(type=Activation, name/shape=(1, 320, 28, 28), dtype=float32)</td>
      <td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.reshape RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 112, 112), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 112, 112), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 56, 56), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 56, 56), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 28, 28), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 192, 28, 28), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 384, 28, 28), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 576, 28, 28), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>clip</td>
      <td>Operand(type=Activation, name/shape=(1, 960, 28, 28), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>concatenate</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(1, 256, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(32, 3, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(16, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96, 16, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(24, 96, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(144, 24, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(144, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(144, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(24, 144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(32, 144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(192, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(32, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(64, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384, 64, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 2<br>groups : 384<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(64, 384, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96, 384, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(576, 96, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(576, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 2<br>groups : 576<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96, 576, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(160, 576, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(960, 160, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(960, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [4, 4, 4, 4]<br>dilation : 4<br>groups : 960<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(160, 960, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(320, 960, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(256, 320, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(256, 320, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(256, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>conv2d</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(21, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>TT_METAL: ttnn.tilize_with_val_padding validation RuntimeError tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_op.cpp input_tensor_a.get_dtype() == DataType::BFLOAT16 or input_tensor_a.get_dtype() == DataType::UINT32 Can only tilize bfloat16 or uint32 tensors</td>
    </tr>
    <tr>
      <td>identity</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_0430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.conv_stem.first_conv.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_2430, dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=mobilenet_v2.conv_stem.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_5430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_6430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.conv_stem.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_8430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_9430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.0.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_11430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.0.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_14430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_15430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(24,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(24,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(24,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(24, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.0.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_17430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(24,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(24,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_18430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(144, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.1.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_20430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.1.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_23430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.1.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_26430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.2.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_29430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(144, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.2.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_32430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.2.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_35430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_36430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.3.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_38430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.3.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_41430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.3.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_44430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.4.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_47430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.4.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_50430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.4.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_53430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.5.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_56430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.5.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_59430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_60430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64,), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.5.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_62430, dtype=float32)</td>
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
      <td>Operand(type=Constant, name/shape=const_63430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.6.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_65430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.6.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_68430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.6.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_71430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.7.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_74430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.7.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_77430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.7.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_80430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.8.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_83430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.8.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_86430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.8.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_89430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.9.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_92430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.9.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_95430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.9.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_98430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_99430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(576,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(576, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.10.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_101430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(576,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.10.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_104430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.10.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_107430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.11.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_110430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.11.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_113430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.11.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_116430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.12.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_119430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.12.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_122430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_123430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.12.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_125430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_126430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(960,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.13.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_128430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(960,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.13.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_131430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.13.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_134430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.14.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_137430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.14.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_140430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.14.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_143430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.15.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_146430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.15.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_149430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_150430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, name/shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(320, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.15.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_152430, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Constant, name/shape=const_153430, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256,), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=segmentation_head.conv_pool.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_155430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
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
      <td>Operand(type=Activation, name/shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=segmentation_head.conv_aspp.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_158430, dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: ttmetal vs Forge Output Data mismatch AssertionError assert False where False = all([False])</td>
    </tr>
    <tr>
      <td>multiply</td>
      <td>Operand(type=Activation, name/shape=segmentation_head.conv_projection.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, name/shape=const_161430, dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(24,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reciprocal</td>
      <td>Operand(type=Activation, name/shape=(320,), dtype=float32)</td>
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
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>relu</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.conv_stem.conv_3x3.convolution.weight, dtype=float32)</td>
      <td>shape : (32, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.0.conv_3x3.convolution.weight, dtype=float32)</td>
      <td>shape : (96, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.1.conv_3x3.convolution.weight, dtype=float32)</td>
      <td>shape : (144, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.3.conv_3x3.convolution.weight, dtype=float32)</td>
      <td>shape : (192, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.6.conv_3x3.convolution.weight, dtype=float32)</td>
      <td>shape : (384, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.10.conv_3x3.convolution.weight, dtype=float32)</td>
      <td>shape : (576, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>reshape</td>
      <td>Operand(type=Activation, name/shape=mobilenet_v2.layer.13.conv_3x3.convolution.weight, dtype=float32)</td>
      <td>shape : (960, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>resize2d</td>
      <td>Operand(type=Activation, name/shape=(1, 256, 1, 1), dtype=float32)</td>
      <td>sizes : [28, 28]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>FORGE: lower_to_mlir RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
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
      <td>Operand(type=Activation, name/shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(24,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>sqrt</td>
      <td>Operand(type=Activation, name/shape=(320,), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(16,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
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
      <td>Operand(type=Activation, name/shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(24,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(24, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(144,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(144, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(192,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(192, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=(384,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(384, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(576,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(576, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(160,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(160, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(960,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(960, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(320,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(320, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, name/shape=segmentation_head.classifier.convolution.bias, dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>MLIR: ttnn.reshape mlir pipeline RuntimeError 'ttnn.reshape' op Shape attribute size must match output tensor rank Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>unsqueeze</td>
      <td>Operand(type=Activation, name/shape=(21, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
