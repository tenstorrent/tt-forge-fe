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
      <td>Operand(type=Activation, shape=(1, 1000), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1000,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(256,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1024, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit1.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit1.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit2.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit3.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit3.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.unit3.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit1.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit1.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit2.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit3.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit3.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit3.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_61830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit4.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit4.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.unit4.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit1.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit1.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit2.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit3.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_94830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit3.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_97830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit3.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit4.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit4.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_106830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit4.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_109830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit5.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit5.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_115830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit5.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_118830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit6.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_121830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit6.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_124830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit6.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_127830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit7.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_130830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit7.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_133830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit7.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_136830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit8.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_139830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit8.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_142830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit8.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_145830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit9.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_148830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit9.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_151830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit9.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_154830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit10.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_157830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit10.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_160830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit10.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_163830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit11.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_166830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit11.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_169830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit11.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_172830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit12.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_175830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit12.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_178830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit12.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_181830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit13.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_184830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit13.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_187830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit13.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_190830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit14.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_193830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit14.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_196830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit14.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_199830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit15.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_202830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit15.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_205830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit15.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_208830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit16.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_211830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit16.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_214830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit16.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_217830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit17.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_220830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit17.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_223830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit17.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_226830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit18.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_229830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit18.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_232830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit18.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_235830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit19.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_238830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit19.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_241830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit19.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_244830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit20.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_247830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit20.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_250830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit20.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_253830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit21.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_256830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit21.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_259830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit21.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_262830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit22.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_265830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit22.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_268830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit22.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_271830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit23.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_274830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit23.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_277830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.unit23.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_280830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_283830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_286830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit1.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_289830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit1.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_292830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_295830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_298830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit2.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_301830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit3.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_304830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit3.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_307830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage4.unit3.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_310830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)</td>
      <td>kernel_size : [7, 7]<br>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.reshape] RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 4, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 8, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 8, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 16, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 16, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 32, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 32, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1024, 1, 1), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1024, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1024, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 2048, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1000), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
      <td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_0965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_39965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(256,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(256,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_114965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_261965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1024, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_291838, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit1.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit1.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit2.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit3.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit3.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.unit3.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit1.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit1.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit2.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit3.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit3.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit3.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_62830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit4.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_65830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit4.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_68830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.unit4.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_71830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_74830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit1.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit1.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit2.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit3.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_95830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit3.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit3.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit4.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit4.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_107830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit4.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit5.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit5.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit5.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit6.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_122830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit6.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit6.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit7.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit7.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_134830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit7.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_137830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit8.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_140830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit8.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_143830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit8.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_146830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit9.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_149830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit9.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_152830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit9.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_155830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit10.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_158830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit10.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_161830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit10.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_164830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit11.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_167830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit11.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_170830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit11.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_173830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit12.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_176830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit12.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_179830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit12.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_182830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit13.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_185830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit13.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_188830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit13.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_191830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit14.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_194830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit14.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_197830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit14.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_200830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit15.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_203830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit15.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_206830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit15.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_209830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit16.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_212830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit16.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_215830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit16.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_218830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit17.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_221830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit17.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_224830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit17.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_227830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit18.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_230830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit18.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_233830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit18.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_236830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit19.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_239830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit19.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_242830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit19.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_245830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit20.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_248830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit20.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_251830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit20.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_254830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit21.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_257830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit21.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_260830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit21.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_263830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit22.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_266830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit22.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_269830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit22.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_272830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit23.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_275830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit23.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_278830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.unit23.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_281830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_284830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_287830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit1.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_290830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit1.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_293830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_296830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_299830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit2.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_302830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit3.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_305830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit3.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_308830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage4.unit3.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_311830, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(2048,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1024, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 2048, 1, 1), dtype=float32)</td>
      <td>shape : (1, 2048)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(64,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(256,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(512,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1024,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(2048,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1000, 2048), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(256, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(64, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(64,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(256,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(512,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(512, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1024,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1024, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2048,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2048, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
