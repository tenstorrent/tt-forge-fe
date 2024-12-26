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
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(224,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stem.stem_1/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stem.stem_2/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stem.stem_3/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.0.OSA2_1_0/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.1.OSA2_1_1/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.2.OSA2_1_2/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.3.OSA2_1_3/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.4.OSA2_1_4/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.concat.OSA2_1_concat/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.0.OSA3_1_0/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.1.OSA3_1_1/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.2.OSA3_1_2/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.3.OSA3_1_3/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.4.OSA3_1_4/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.concat.OSA3_1_concat/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.0.OSA4_1_0/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.1.OSA4_1_1/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.2.OSA4_1_2/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.3.OSA4_1_3/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.4.OSA4_1_4/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.concat.OSA4_1_concat/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_61310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.0.OSA4_2_0/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.1.OSA4_2_1/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.2.OSA4_2_2/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.3.OSA4_2_3/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.4.OSA4_2_4/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.concat.OSA4_2_concat/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.0.OSA5_1_0/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.1.OSA5_1_1/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.2.OSA5_1_2/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.3.OSA5_1_3/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.4.OSA5_1_4/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_94310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.concat.OSA5_1_concat/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_97310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.0.OSA5_2_0/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.1.OSA5_2_1/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.2.OSA5_2_2/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_106310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.3.OSA5_2_3/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_109310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.4.OSA5_2_4/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.concat.OSA5_2_concat/norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_115310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)</td>
      <td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.reshape] RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 768, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 256, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 160, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1056, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1056, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 512, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1472, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 1472, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 768, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1728, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 1728, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 768, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 224, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1888, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1888, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 1024, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 2144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1000), dtype=float32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td></td>
    </tr>
    <tr>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
      <td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.maxpool2d mlir pipeline] RuntimeError ttnn.max_pool2d currently only supports an input type of bfloat16 Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
      <td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.maxpool2d mlir pipeline] RuntimeError ttnn.max_pool2d currently only supports an input type of bfloat16 Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)</td>
      <td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][ttnn.maxpool2d mlir pipeline] RuntimeError ttnn.max_pool2d currently only supports an input type of bfloat16 Failed to run MLIR compiler pass pipeline</td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_01605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_61605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_211605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_271605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_331605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(224,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(224,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_391605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1141605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_2131605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2611605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stem.stem_1/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stem.stem_2/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stem.stem_3/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.0.OSA2_1_0/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.1.OSA2_1_1/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.2.OSA2_1_2/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.3.OSA2_1_3/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.layers.4.OSA2_1_4/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage2.OSA2_1.concat.OSA2_1_concat/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.0.OSA3_1_0/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.1.OSA3_1_1/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.2.OSA3_1_2/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.3.OSA3_1_3/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.layers.4.OSA3_1_4/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage3.OSA3_1.concat.OSA3_1_concat/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.0.OSA4_1_0/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.1.OSA4_1_1/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.2.OSA4_1_2/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.3.OSA4_1_3/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.layers.4.OSA4_1_4/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_1.concat.OSA4_1_concat/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_62310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.0.OSA4_2_0/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_65310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.1.OSA4_2_1/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_68310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.2.OSA4_2_2/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_71310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.3.OSA4_2_3/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_74310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.layers.4.OSA4_2_4/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage4.OSA4_2.concat.OSA4_2_concat/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.0.OSA5_1_0/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.1.OSA5_1_1/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.2.OSA5_1_2/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.3.OSA5_1_3/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.layers.4.OSA5_1_4/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_95310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_1.concat.OSA5_1_concat/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.0.OSA5_2_0/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.1.OSA5_2_1/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.2.OSA5_2_2/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_107310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.3.OSA5_2_3/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.layers.4.OSA5_2_4/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113310, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=stage5.OSA5_2.concat.OSA5_2_concat/norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116310, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(224,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(768,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)</td>
      <td>shape : (1, 1024)</td>
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
      <td>Operand(type=Activation, shape=(128,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(192,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(224,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(768,), dtype=float32)</td>
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
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1000, 1024), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(768, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192, 1), dtype=float32)</td>
      <td>dim : 1</td>
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
      <td>Operand(type=Activation, shape=(128,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(128, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(160,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(160, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(192,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(224,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(224, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(768,), dtype=float32)</td>
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
  </tbody>
</table>
