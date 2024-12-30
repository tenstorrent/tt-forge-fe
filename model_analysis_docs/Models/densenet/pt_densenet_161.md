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
      <td>Operand(type=Constant, name=features.norm0.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer1.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer1.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_71605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer2.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer2.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer3.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_161605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer3.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_191605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer4.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_221605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer4.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_251605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer5.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_281605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer5.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_311605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer6.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_341605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer6.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_371605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.transition1.norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_401605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer1.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_431605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer1.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_461605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer2.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_491605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer2.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_521605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer3.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_551605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer3.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_581605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer4.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_611605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer4.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_641605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer5.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_671605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer5.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_701605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer6.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_731605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer6.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_761605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer7.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_791605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer7.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_821605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer8.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_851605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer8.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_881605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer9.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_911605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer9.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_941605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer10.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_971605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer10.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1001605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer11.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1031605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer11.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1061605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer12.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1091605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer12.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1121605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.transition2.norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1151605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer1.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1181605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer1.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1211605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer2.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1241605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer2.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1271605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer3.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1301605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer3.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1331605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer4.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1361605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer4.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1391605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer5.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1421605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer5.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1451605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer6.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1481605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer6.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1511605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer7.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1541605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer7.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1571605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer8.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1601605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer8.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1631605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer9.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1661605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer9.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1691605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer10.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1721605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer10.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1751605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer11.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1781605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer11.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1811605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer12.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1841605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer12.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1871605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer13.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1901605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer13.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1931605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer14.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1961605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer14.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1991605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer15.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2021605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer15.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2051605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer16.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2081605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer16.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2111605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer17.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2141605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer17.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2171605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer18.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2201605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer18.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2231605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer19.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2261605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer19.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2291605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer20.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2321605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer20.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2351605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer21.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2381605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer21.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2411605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer22.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2441605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer22.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2471605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer23.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2501605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer23.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2531605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer24.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2561605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer24.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2591605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer33.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3101605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer33.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3131605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer34.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3161605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer34.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3191605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer35.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3221605, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1344, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer35.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3251605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer36.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3281605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer36.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3311605, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1440, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1440, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1536, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1632, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1728,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1728,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1728, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1728, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.transition3.norm.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4061605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer1.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4091605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer1.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4121605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer2.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4151605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer2.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4181605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer3.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4211605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer3.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4241605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer4.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4271605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer4.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4301605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer5.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4331605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer5.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4361605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer6.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4391605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer6.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4421605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer7.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4451605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer7.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4481605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer8.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4511605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer8.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4541605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer9.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4571605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer9.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4601605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer10.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4631605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer10.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4661605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer11.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4691605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer11.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4721605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer12.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4751605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer12.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4781605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer13.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4811605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer13.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4841605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer14.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4871605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer14.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4901605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer15.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4931605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer15.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4961605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer16.norm1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4991605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer16.norm2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5021605, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1728, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1728, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1824,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1824,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1824, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1824, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.norm5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6011605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1920,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1920,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1920, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1920, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(240,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 240, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 288, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(336,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 384, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(432,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 432, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(528,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 528, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(528, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(624,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(624,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 624, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(624, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(720,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 720, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 768, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(528, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 624, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(624, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(816,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(816,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 816, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(816, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(912,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(912,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 912, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(912, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1008,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1008,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1008, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1008, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1104,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1104,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1104, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1104, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1200,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1200, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1296,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1296,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1296, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1296, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1392,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1392, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1488,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1488,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1488, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1488, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1584,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1584,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1584, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1584, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1680,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1680,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1680, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1680, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1776,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1776,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1776, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1776, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1824, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1824, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1872,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1872,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1872, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1872, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1920, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1920, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1968,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1968,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1968, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1968, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(2016,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2016,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2016, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2016, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(2064,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2064,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2064, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2064, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(2112,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2112,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2112, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 192, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1104, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1104, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1200, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1296, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1296, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1392, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1392, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1488, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1488, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1584, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1584, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1680, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1680, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1776, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1776, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1872, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1872, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1968, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1968, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2016, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2016, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2064, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2064, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2112, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(2160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(2208,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2208,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2208, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2208, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 2208, 7, 7), dtype=float32)</td>
      <td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.reshape] RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 3, 7, 7), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 96, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 240, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 288, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 288, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 336, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 240, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 288, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 336, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 432, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 432, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 480, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 528, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 528, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 576, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 624, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 624, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 672, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 720, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 720, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 768, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 432, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 480, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 528, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 576, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 624, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 624, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 672, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 720, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 768, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 816, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 816, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 864, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 864, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 912, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 912, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 960, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1008, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1008, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1056, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1104, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1104, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1152, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1152, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1200, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1248, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1248, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1296, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1296, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1344, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1392, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1440, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1440, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1488, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1488, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1536, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1536, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1584, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1584, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1632, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1632, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1680, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1680, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1728, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1728, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1776, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1776, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1824, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1824, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1872, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1872, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1920, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1920, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1968, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1968, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2016, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2016, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2064, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2064, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 2112, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1056, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1104, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1104, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1152, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1200, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1248, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1248, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1296, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1296, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1344, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1344, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1392, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1392, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1440, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1488, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1488, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1536, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1536, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1584, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1584, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1632, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1680, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1680, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1728, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1728, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1776, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1776, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1824, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1824, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1872, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1872, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1920, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1920, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1968, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1968, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2016, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2016, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2064, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2064, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2112, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 2160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2160, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 2208), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2208, 1000), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)</td>
      <td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.norm0.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_21605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer1.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_51605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer1.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_81605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_91605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer2.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_111605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer2.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_141605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer3.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_171605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer3.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_201605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer4.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_231605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer4.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_261605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer5.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_291605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock1.denselayer5.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_321605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer6.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_351605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock1.denselayer6.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_381605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.transition1.norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_411605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer1.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_441605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer1.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_471605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer2.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_501605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer2.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_531605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer3.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_561605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer3.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_591605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer4.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_621605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer4.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_651605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer5.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_681605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer5.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_711605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_721605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer6.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_741605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer6.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_771605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer7.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_801605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer7.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_831605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer8.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_861605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer8.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_891605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_901605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer9.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_921605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer9.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_951605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer10.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_981605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer10.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1011605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer11.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1041605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock2.denselayer11.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1071605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1081605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer12.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1101605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock2.denselayer12.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1131605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.transition2.norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1161605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer1.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1191605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer1.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1221605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer2.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1251605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer2.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1281605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer3.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1311605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer3.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1341605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer4.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1371605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer4.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1401605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer5.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1431605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer5.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1461605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer6.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1491605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer6.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1521605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer7.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1551605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer7.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1581605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer8.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1611605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer8.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1641605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer9.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1671605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer9.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1701605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer10.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1731605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer10.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1761605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1771605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer11.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1791605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer11.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1821605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer12.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1851605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer12.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1881605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer13.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1911605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer13.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1941605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1951605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer14.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1971605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer14.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2001605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer15.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2031605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer15.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2061605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer16.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2091605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer16.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2121605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer17.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2151605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer17.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2181605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer18.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2211605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer18.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2241605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer19.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2271605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer19.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2301605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2311605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(864,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer20.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2331605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer20.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2361605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer21.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2391605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer21.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2421605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer22.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2451605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer22.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2481605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2491605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer23.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2511605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer23.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2541605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer24.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2571605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer24.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2601605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer33.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3111605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer33.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3141605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer34.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3171605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer34.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3201605, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1344, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer35.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3231605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock3.denselayer35.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3261605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer36.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3291605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock3.denselayer36.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3321605, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1440, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1440, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1536, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1632, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_3931605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1728,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1728,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1728,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1728, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1728, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1728,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1728,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.transition3.norm.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4071605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer1.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4101605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer1.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4131605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer2.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4161605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer2.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4191605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer3.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4221605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer3.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4251605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer4.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4281605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer4.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4311605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer5.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4341605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer5.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4371605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer6.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4401605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer6.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4431605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer7.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4461605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer7.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4491605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer8.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4521605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer8.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4551605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer9.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4581605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer9.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4611605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer10.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4641605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer10.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4671605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer11.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4701605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer11.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4731605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer12.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4761605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer12.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4791605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer13.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4821605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer13.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4851605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer14.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4881605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer14.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4911605, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.denseblock4.denselayer15.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4941605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer15.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4971605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer16.norm1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5001605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.denseblock4.denselayer16.norm2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5031605, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1728, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1728, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_5821605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1824,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1824,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1824,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1824, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1824, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1824,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1824,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_6001605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1920,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1920,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1920,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1920, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1920, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.norm5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6021605, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1920,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1920,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_91285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_211285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(240,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 240, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(240,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 288, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_331285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(336,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(336,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 384, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_721285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(432,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 432, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(432,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_841285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(528,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(528,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 528, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(528, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(528,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(528,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_961285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(624,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(624,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(624,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 624, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(624, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(624,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(624,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1081285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(720,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 720, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(720,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 768, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(528, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 624, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(624, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1711285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(816,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(816,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(816,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 816, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(816, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(816,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(816,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1831285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(912,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(912,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(912,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 912, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(912, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(912,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(912,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1951285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1008,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1008,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1008,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1008, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1008, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1008,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1008,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2071285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1104,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1104,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1104,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1104, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1104, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1104,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1104,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2191285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1200,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1200, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1200,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2311285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1296,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1296,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1296,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1296, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1296, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1296,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1296,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2431285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1392,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1392,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1392, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1392,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1392,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2551285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1488,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1488,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1488,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1488, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1488, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1488,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1488,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2671285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1584,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1584,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1584,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1584, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1584, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1584,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1584,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2791285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1680,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1680,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1680,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1680, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1680, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1680,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1680,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_2911285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1776,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1776,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1776,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1776, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1776, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1776,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1776,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1824, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1824, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3031285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1872,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1872,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1872,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1872, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1872, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1872,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1872,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1920, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1920, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3151285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1968,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1968,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1968,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1968, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1968, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1968,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1968,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3211285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2016,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2016,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2016,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2016, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2016, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2016,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2016,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3271285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2064,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2064,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2064,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2064, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2064, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2064,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2064,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_3331285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2112,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2112,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2112,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2112, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2112,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2112,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 192, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1104, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1104, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1200, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1296, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1296, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1392, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1392, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1488, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1488, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1584, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1584, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1680, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1680, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1776, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1776, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1872, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1872, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1968, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1968, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2016, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2016, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2064, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2064, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2112, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_4741285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_4801285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2208,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2208,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2208,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 2208, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2208, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(2208,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2208,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1728,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1824,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1920,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(240,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(336,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(432,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(528,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(624,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(720,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(816,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(912,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1008,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1104,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1296,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1392,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1488,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1584,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1680,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1776,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1872,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1968,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(2016,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(2064,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(2112,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(2160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(2208,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1440, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1536, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1632, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1728, 14, 14), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1728, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1824, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1920, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 240, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 288, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 384, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 432, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 528, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 624, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 720, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 768, 28, 28), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 624, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 816, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 912, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1008, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1104, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1200, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1296, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1488, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1584, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1680, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1776, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1824, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1872, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1920, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1968, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2016, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2064, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2112, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 192, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1104, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1200, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1296, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1392, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1488, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1584, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1680, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1776, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1872, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1968, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2016, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2064, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2112, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2160, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 2208, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 2208, 1, 1), dtype=float32)</td>
      <td>shape : (1, 2208, 1, 1)</td>
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
      <td>Operand(type=Activation, shape=(1728,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1824,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1920,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(144,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(240,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(336,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(432,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(528,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(624,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(720,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(816,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(912,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1008,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1104,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1296,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1392,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1488,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1584,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1680,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1776,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1872,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1968,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(2016,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(2064,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(2112,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(2160,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(2208,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 2208, 1, 1), dtype=float32)</td>
      <td>dim : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 2208, 1), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1000, 2208), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1728,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1728, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1824,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1824, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1920,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1920, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(144,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(144, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(240,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(240, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(336,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(336, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(432,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(432, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(528,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(528, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(624,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(624, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(720,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(720, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(816,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(816, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(912,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(912, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1008,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1008, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1104,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1104, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1200,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1200, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1296,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1296, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1392,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1392, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1488,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1488, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1584,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1584, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1680,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1680, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1776,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1776, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1872,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1872, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1968,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1968, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2016,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2016, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2064,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2064, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2112,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2112, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2160,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2160, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2208,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2208, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
