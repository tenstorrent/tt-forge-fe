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
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.norm0.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer1.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer1.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer2.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer2.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer3.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer3.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer4.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22965, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 160, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer4.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer5.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28965, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer5.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer6.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34965, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer6.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.transition1.norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer1.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer1.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer2.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer2.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer3.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer3.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer4.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_61965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 224, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer4.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer5.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer5.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer6.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(288,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer6.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer7.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer7.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer8.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(352,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(352,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 352, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(352, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer8.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer9.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer9.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_94965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer10.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_97965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(416,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(416,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 416, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(416, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer10.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer11.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(448,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer11.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_106965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer12.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_109965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(480,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer12.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.transition2.norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_115965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer1.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_118965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer1.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_121965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer2.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_124965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer2.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_127965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer3.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_130965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer3.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_133965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer4.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_136965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 352, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(352, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer4.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_139965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer5.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_142965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer5.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_145965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer6.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_148965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 416, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(416, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer6.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_151965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer7.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_154965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer7.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_157965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer8.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_160965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer8.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_163965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer9.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_166965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer9.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_169965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer10.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_172965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(544,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(544,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 544, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(544, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer10.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_175965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer11.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_178965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(576,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer11.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_181965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer12.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_184965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(608,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(608,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 608, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(608, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer12.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_187965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer13.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_190965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(640,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(640,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 640, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(640, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer13.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_193965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer14.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_196965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(672,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer14.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_199965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer15.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_202965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(704,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(704,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 704, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(704, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer15.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_205965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer16.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_208965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(736,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(736,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 736, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(736, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer16.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_211965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer17.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_214965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer17.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_217965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer18.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_220965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(800,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(800,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 800, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(800, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer18.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_223965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer19.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_226965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(832,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(832,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 832, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(832, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer19.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_229965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer20.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_232965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(864,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(864,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 864, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(864, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer20.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_235965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer21.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_238965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(896,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer21.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_241965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer22.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_244965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(928,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(928,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 928, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(928, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer22.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_247965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer23.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_250965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(960,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer23.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_253965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer24.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_256965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(992,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(992,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 992, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(992, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer24.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_259965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.transition3.norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_262965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer1.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_265965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer1.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_268965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer2.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_271965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer2.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_274965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer3.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_277965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer3.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_280965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer4.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_283965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer4.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_286965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer5.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_289965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(640, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer5.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_292965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer6.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_295965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer6.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_298965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer7.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_301965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 704, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(704, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer7.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_304965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer8.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_307965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 736, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(736, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer8.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_310965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer9.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_313965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer9.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_316965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer10.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_319965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 800, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(800, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer10.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_322965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer11.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_325965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(832, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer11.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_328965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer12.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_331965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 864, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(864, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer12.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_334965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer13.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_337965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer13.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_340965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer14.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_343965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 928, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(928, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer14.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_346965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer15.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_349965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer15.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_352965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer16.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_355965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 992, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(992, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer16.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_358965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.norm5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_361965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer25.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2621605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer25.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2651605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer26.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2681605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1056,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1056, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer26.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2711605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer27.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2741605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer27.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2771605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer28.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2801605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1120,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer28.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2831605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer29.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2861605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1152,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1152,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1152, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer29.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2891605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer30.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2921605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1184,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1184, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer30.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2951605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer31.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2981605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1216,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1216,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1216, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1216, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer31.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3011605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer32.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3041605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1248,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1248,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1248, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1248, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer32.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3071605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1280, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1312,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1312,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1344,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1344,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1376,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1376,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1408,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1408,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1440,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1440,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1472,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1472,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1504,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1504,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1568,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1568,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1600,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1600,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1632,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1632,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1664,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1664,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1056, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1120, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1184, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1216, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1216, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1248, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1248, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1312, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1312, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1344, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1344, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1376, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1376, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer17.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5051605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1408, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1408, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer17.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5081605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer18.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5111605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1440, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer18.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5141605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer19.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5171605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1472, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1472, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer19.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5201605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer20.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5231605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1504, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1504, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer20.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5261605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer21.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5291605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1536, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer21.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5321605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer22.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5351605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1568, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1568, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer22.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5381605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer23.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5411605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1600, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1600, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer23.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5441605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer24.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5471605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer24.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5501605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer25.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5531605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1664, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1664, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer25.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5561605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer26.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5591605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer26.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5621605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer27.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5651605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer27.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5681605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer28.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5711605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer28.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5741605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer29.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5771605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer29.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5801605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer30.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5831605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer30.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5861605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer31.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5891605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer31.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5921605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer32.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5951605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer32.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5981605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 640, 14, 14), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 1664, 7, 7), dtype=float32)</td>
      <td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.reshape] RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
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
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 96, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 160, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 160, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 224, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 160, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 224, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 224, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 288, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 320, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 352, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 352, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 384, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 416, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 416, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 448, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 480, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 288, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 320, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 352, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 352, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 384, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 416, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 416, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 448, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 480, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 544, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 544, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 576, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 608, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 608, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 640, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 640, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 672, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 704, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 704, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 736, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 736, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 768, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 800, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 800, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 832, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 832, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 864, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 864, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 896, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 928, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 928, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 960, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 992, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 992, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 640, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 672, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 704, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 704, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 736, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 736, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 768, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 800, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 800, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 832, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 864, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 864, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 896, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 928, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 928, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 960, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 992, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 992, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1024, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1056, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1088, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1120, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1152, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1152, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1184, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1216, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1216, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1248, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1248, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1024, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1056, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1088, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1120, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1120, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1152, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1184, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1216, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1216, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1248, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1248, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1280, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1312, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1312, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1344, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1344, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1376, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1376, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1408, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1408, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1440, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1472, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1472, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1504, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1504, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1536, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1536, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1568, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1568, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1600, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1600, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1632, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1280, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(640, 1280, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1664), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1664, 1000), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.norm0.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2965, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer1.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_6965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer1.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_9965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer2.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer2.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer3.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer3.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_21965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 160, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer4.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer4.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_27965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer5.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer5.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_33965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer6.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer6.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.transition1.norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41965, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer1.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer1.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer2.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer2.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer3.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer3.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 224, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer4.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_62965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer4.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_65965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer5.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_68965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer5.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_71965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_72965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(288,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer6.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_74965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(288,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer6.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_78965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer7.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer7.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_84965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(352,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(352,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(352,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 352, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(352, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer8.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(352,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(352,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer8.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_90965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer9.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer9.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_95965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_96965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(416,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(416,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(416,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 416, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(416, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer10.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(416,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(416,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer10.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_102965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(448,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer11.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(448,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer11.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_107965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_108965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(480,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer12.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(480,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer12.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.transition2.norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116965, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer1.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer1.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_122965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer2.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer2.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer3.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer3.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_134965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 352, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(352, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer4.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_137965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer4.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_140965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer5.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_143965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer5.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_146965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 416, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(416, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer6.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_149965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer6.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_152965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer7.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_155965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer7.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_158965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer8.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_161965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer8.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_164965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer9.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_167965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer9.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_170965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_171965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(544,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(544,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(544,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 544, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(544, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer10.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_173965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(544,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(544,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer10.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_176965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_177965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(576,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer11.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_179965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(576,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer11.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_182965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_183965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(608,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(608,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(608,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 608, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(608, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer12.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_185965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(608,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(608,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer12.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_188965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_189965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(640,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(640,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(640,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 640, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(640, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer13.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_191965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(640,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(640,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer13.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_194965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_195965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(672,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer14.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_197965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(672,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer14.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_200965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_201965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(704,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(704,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(704,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 704, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(704, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer15.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_203965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(704,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(704,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer15.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_206965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_207965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(736,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(736,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(736,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 736, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(736, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer16.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_209965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(736,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(736,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer16.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_212965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_213965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer17.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_215965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer17.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_218965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_219965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(800,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(800,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(800,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 800, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(800, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer18.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_221965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(800,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(800,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer18.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_224965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_225965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(832,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(832,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(832,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 832, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(832, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer19.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_227965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(832,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(832,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer19.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_230965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_231965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(864,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(864,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(864,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 864, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(864, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer20.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_233965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(864,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(864,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer20.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_236965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_237965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(896,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer21.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_239965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(896,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer21.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_242965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_243965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(928,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(928,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(928,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 928, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(928, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer22.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_245965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(928,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(928,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer22.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_248965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_249965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(960,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer23.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_251965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(960,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer23.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_254965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_255965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(992,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(992,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(992,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 992, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(992, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer24.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_257965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(992,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(992,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer24.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_260965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.transition3.norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_263965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer1.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_266965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer1.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_269965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer2.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_272965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer2.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_275965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer3.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_278965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer3.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_281965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer4.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_284965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer4.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_287965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(640, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer5.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_290965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer5.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_293965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer6.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_296965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer6.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_299965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 704, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(704, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer7.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_302965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer7.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_305965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 736, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(736, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer8.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_308965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer8.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_311965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer9.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_314965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer9.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_317965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 800, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(800, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer10.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_320965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer10.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_323965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(832, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer11.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_326965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer11.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_329965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 864, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(864, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer12.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_332965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer12.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_335965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer13.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_338965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer13.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_341965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 928, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(928, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer14.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_344965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer14.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_347965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer15.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_350965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer15.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_353965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 992, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(992, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer16.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_356965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer16.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_359965, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.norm5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_362965, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer25.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2631605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer25.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2661605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2671605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1056,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1056,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1056, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer26.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2691605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1056,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1056,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer26.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2721605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2731605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer27.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2751605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer27.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2781605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2791605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1120,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer28.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2811605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1120,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer28.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2841605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2851605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1152,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1152,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1152, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer29.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2871605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1152,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer29.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2901605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2911605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1184,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1184, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer30.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2931605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1184,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer30.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2961605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2971605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1216,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1216,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1216,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1216, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1216, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer31.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2991605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1216,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1216,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer31.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3021605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3031605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1248,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1248,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1248,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1248, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1248, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer32.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3051605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1248,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1248,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer32.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3081605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3091605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1280, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3151605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1312,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1312,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1312,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1312,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1312,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3211605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1344,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1344,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1344,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1344,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1344,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3271605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1376,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1376,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1376,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1376,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1376,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3331605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1408,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1408,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1408,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1408,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1408,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3391605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1440,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1440,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1440,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1440,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1440,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3451605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1472,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1472,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1472,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1472,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1472,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3511605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1504,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1504,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1504,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1504,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1504,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3571605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3631605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1568,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1568,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1568,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1568,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1568,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3691605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1600,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1600,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1600,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1600,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1600,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3751605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1632,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1632,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1632,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3811605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1664,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1664,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1664,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1664,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1664,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1056, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1120, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1184, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1216, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1216, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1248, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1248, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1312, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1312, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1344, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1344, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1376, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1376, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1408, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1408, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer17.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5061605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer17.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5091605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1440, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer18.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5121605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer18.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5151605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1472, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1472, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer19.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5181605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer19.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5211605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1504, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1504, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer20.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5241605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer20.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5271605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1536, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer21.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5301605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer21.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5331605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1568, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1568, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer22.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5361605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer22.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5391605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1600, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1600, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer23.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5421605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer23.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5451605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer24.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5481605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer24.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5511605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1664, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1664, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer25.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5541605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer25.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5571605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer26.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5601605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer26.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5631605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer27.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5661605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer27.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5691605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer28.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5721605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer28.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5751605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer29.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5781605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer29.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5811605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer30.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5841605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer30.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5871605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer31.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5901605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer31.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5931605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer32.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5961605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer32.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5991605, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(288,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(352,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(416,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(448,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(480,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(544,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(608,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(640,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(672,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(704,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(736,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(800,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(832,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(864,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(896,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(928,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(992,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1056,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1152,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1216,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1248,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1312,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1344,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1376,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1408,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1440,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1472,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1504,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1536,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1568,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1600,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1632,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1664,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 160, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 224, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 352, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 416, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 352, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 416, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 544, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 608, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 640, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 704, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 736, 14, 14), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 800, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 832, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 864, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 928, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 992, 14, 14), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 704, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 736, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 800, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 864, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 928, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 992, 7, 7), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1120, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1152, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1184, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1216, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1248, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1280, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1120, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1184, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1216, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1248, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1312, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1344, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1376, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1408, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1440, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1472, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1504, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1536, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1568, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1600, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1664, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1664, 1, 1), dtype=float32)</td>
      <td>shape : (1, 1664, 1, 1)</td>
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
      <td>Operand(type=Activation, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(288,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(320,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(352,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(384,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(416,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(448,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(480,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(544,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(576,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(608,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(640,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(672,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(704,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(736,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(800,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(832,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(864,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(896,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(928,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(992,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1056,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1152,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1216,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1248,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1312,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1344,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1376,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1408,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1440,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1472,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1504,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1536,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1568,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1600,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1632,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1664,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 1664, 1, 1), dtype=float32)</td>
      <td>dim : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 1664, 1), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1000, 1664), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(96, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(384, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(288,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(288, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(320,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(320, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(352,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(352, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(384,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(416,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(416, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(448,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(448, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(480,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(480, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(544,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(544, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(576,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(576, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(608,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(608, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(640,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(640, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(672,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(672, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(704,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(704, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(736,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(736, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(800,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(800, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(832,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(832, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(864,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(864, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(896,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(896, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(928,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(928, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(960,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(960, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(992,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(992, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1056,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1056, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1088, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1120,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1120, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1152,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1152, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1184,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1184, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1216,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1216, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1248,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1248, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1280, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1312,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1312, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1344,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1344, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1376,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1376, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1408,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1408, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1440,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1440, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1472,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1472, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1504,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1504, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1536,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1536, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1568,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1568, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1600,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1600, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1632,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1632, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1664,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1664, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
