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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(36,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(72,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_12602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_42602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_72602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_102602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block1.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_132602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block1.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_162602, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=features.init_block.subblocks.block2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_192602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_222602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block2.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_252602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.transition.block1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_462602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch1.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_492602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch1.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_522602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch1.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_552602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch1.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_582602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.transition.block2.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_732602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch2.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_762602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch2.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_792602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch2.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_822602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch2.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_852602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.fuse_layers.layer1.block2.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1002602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch1.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1032602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch1.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1062602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch1.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1092602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch1.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1122602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.fuse_layers.layer2.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1272602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch2.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1302602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch2.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1332602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch2.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1362602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch2.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1392602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer1.block2.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1542602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.transition.block3.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1572602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch3.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1602602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch3.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1632602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch3.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1662602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch3.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1692602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer1.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1842602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch1.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1872602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch1.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1902602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch1.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1932602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch1.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1962602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer2.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2112602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer2.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2142602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch2.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2172602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch2.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2202602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch2.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2232602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch2.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2262602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer1.block2.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2412602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer3.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2442602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer3.block1.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2472602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer3.block2.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2502602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch3.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2532602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch3.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2562602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch3.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2592602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch3.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2622602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer1.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2772602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch1.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2802602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch1.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2832602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch1.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2862602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch1.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2892602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer2.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3042602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer2.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3072602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch2.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3102602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch2.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3132602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch2.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3162602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch2.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3192602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer1.block2.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3342602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer3.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3372602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer3.block1.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3402602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer3.block2.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3432602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch3.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3462602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch3.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3492602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch3.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3522602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch3.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3552602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer1.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3702602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer2.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3972602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer2.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4002602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer3.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4302602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer3.block1.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4332602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer3.block2.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4362602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch1.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4662602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch1.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4692602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch1.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4722602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch1.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4752602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch2.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4962602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch2.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4992602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch2.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5022602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch2.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5052602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer1.block2.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5202602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch3.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5322602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch3.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5352602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch3.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5382602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch3.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5412602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer1.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5562602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.transition.block4.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5592602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch4.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5622602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch4.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5652602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch4.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5682602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch4.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5712602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer1.block4.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5862602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer2.block4.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5882602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer3.block4.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5902602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch1.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5952602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch1.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5982602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch1.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6012602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch1.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6042602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer2.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6192602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer3.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6212602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6232602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer2.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6282602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch2.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6312602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch2.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6342602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch2.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6372602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch2.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6402602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer1.block2.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6552602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer3.block1.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6582602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer3.block2.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6612602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch3.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6642602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch3.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6672602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch3.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6702602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch3.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6732602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer1.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6882602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block1.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6912602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block1.subblock3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6942602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block2.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6972602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block2.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7002602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block3.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7032602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch4.unit1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7062602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch4.unit1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7092602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch4.unit2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7122602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch4.unit2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7152602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer1.block4.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7302602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer2.block4.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7322602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer3.block4.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7342602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer2.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7782602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer3.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7802602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block1.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7822602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer2.block3.conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7872602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer3.block1.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8202602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer3.block2.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8232602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block1.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8532602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block1.subblock3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8562602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block2.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8592602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block2.subblock2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8622602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block3.subblock1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8652602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block4.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8922602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block4.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8952602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block4.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8982602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block4.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9012602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block3.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9192602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block3.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9222602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block3.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9252602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block3.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9282602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block2.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9342602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block2.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9372602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block2.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9402602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block2.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9432602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block1.body.conv1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9522602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block1.body.conv2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9552602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block1.body.conv3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9582602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block1.identity_conv.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9612602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.down_blocks.block1.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9642602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.down_blocks.block2.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9672602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.down_blocks.block3.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9702602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Constant, name=features.final_block.final_layer.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9732602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(18,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 18, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 36, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(126, 1, 1), dtype=float32)</td>
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
      <td>Concatenate</td>
      <td>Operand(type=Parameter, shape=(18, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 144, 1, 1), dtype=float32)</td>
      <td>axis : -4</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 18, 3, 3), dtype=float32)</td>
      <td>axis : -4</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Parameter, shape=(36, 18, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)</td>
      <td>axis : -4</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Parameter, shape=(72, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 144, 1, 1), dtype=float32)</td>
      <td>axis : -4</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
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
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1024, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 256, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 256, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 36, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 36, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 36, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 18, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 36, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 72, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 72, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 72, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 18, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 72, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 144, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(126, 144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 18, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 18, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 36, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 72, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 72, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 36, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 36, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 18, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 18, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 18<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
      <td>dim : -3<br>start : 18<br>stop : 54<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
      <td>dim : -3<br>start : 54<br>stop : 126<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 72<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
      <td>dim : -3<br>start : 72<br>stop : 108<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
      <td>dim : -3<br>start : 108<br>stop : 126<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 18<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 36<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td>dim : -3<br>start : 36<br>stop : 54<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td>dim : -3<br>start : 54<br>stop : 72<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td>dim : -3<br>start : 18<br>stop : 36<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td>dim : -3<br>start : 36<br>stop : 72<br>stride : 1</td>
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
      <td>Operand(type=Constant, name=const_61342, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_36680, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(36,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(36,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_54680, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(72,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(72,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block1.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_142602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block1.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_172602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_202602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_232602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.init_block.subblocks.block2.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_262602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.transition.block1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_472602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch1.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_502602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch1.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_532602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch1.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_562602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch1.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_592602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.transition.block2.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_742602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch2.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_772602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch2.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_802602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch2.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_832602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.branches.branch2.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_862602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.fuse_layers.layer1.block2.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1012602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch1.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1042602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch1.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1072602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch1.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1102602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch1.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1132602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage1.layers.block1.fuse_layers.layer2.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1282602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch2.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1312602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch2.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1342602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch2.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1372602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch2.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1402602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer1.block2.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1552602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.transition.block3.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1582602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch3.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1612602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch3.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1642602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch3.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1672602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.branches.branch3.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1702602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer1.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1852602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch1.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1882602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch1.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1912602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch1.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1942602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch1.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1972602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer2.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2122602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer2.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2152602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch2.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2182602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch2.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2212602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch2.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2242602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch2.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2272602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer1.block2.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2422602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer3.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2452602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer3.block1.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2482602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block1.fuse_layers.layer3.block2.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2512602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch3.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2542602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch3.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2572602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch3.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2602602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.branches.branch3.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2632602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer1.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2782602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch1.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2812602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch1.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2842602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch1.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2872602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch1.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2902602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer2.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3052602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer2.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3082602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch2.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3112602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch2.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3142602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch2.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3172602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch2.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3202602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer1.block2.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3352602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer3.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3382602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer3.block1.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3412602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block2.fuse_layers.layer3.block2.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3442602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch3.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3472602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch3.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3502602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch3.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3532602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.branches.branch3.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3562602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer1.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3712602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer2.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3982602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer2.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4012602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer3.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4312602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer3.block1.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4342602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage2.layers.block3.fuse_layers.layer3.block2.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4372602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch1.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4672602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch1.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4702602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch1.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4732602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch1.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4762602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch2.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4972602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch2.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5002602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch2.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5032602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch2.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5062602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer1.block2.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5212602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch3.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5332602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch3.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5362602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch3.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5392602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch3.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5422602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer1.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5572602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.transition.block4.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5602602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch4.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5632602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch4.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5662602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch4.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5692602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.branches.branch4.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5722602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer1.block4.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5912602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer2.block4.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5922602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer3.block4.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5932602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch1.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5962602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch1.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5992602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch1.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6022602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch1.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6052602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer2.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6242602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer3.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6252602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6262602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer2.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6292602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch2.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6322602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch2.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6352602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch2.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6382602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch2.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6412602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer1.block2.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6562602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer3.block1.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6592602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer3.block2.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6622602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch3.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6652602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch3.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6682602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch3.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6712602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch3.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6742602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer1.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6892602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block1.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6922602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block1.subblock3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6952602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block2.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6982602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block2.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7012602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block1.fuse_layers.layer4.block3.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7042602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch4.unit1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7072602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch4.unit1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7102602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch4.unit2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7132602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.branches.branch4.unit2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7162602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer1.block4.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7352602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer2.block4.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7362602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer3.block4.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7372602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer2.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7832602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer3.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7842602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block1.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7852602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer2.block3.conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7882602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer3.block1.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8212602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer3.block2.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8242602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block1.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8542602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block1.subblock3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8572602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block2.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8602602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block2.subblock2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8632602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.stage3.layers.block2.fuse_layers.layer4.block3.subblock1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8662602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block4.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8932602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block4.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8962602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block4.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8992602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block4.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9022602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block3.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9202602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block3.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9232602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block3.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9262602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block3.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9292602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block2.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9352602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block2.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9382602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block2.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9412602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block2.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9442602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block1.body.conv1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9532602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block1.body.conv2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9562602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block1.body.conv3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9592602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.inc_blocks.block1.identity_conv.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9622602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.down_blocks.block1.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9652602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.down_blocks.block2.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9682602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.down_blocks.block3.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9712602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=features.final_block.final_layer.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9742602, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_271314, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(18,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(18,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 18, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 36, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(126, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)</td>
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
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(36,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(72,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(18,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
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
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 18, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 36, 14, 14), dtype=float32)</td>
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
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)</td>
      <td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 14, 14), dtype=float32)</td>
      <td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 14, 14), dtype=float32)</td>
      <td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 18, 7, 7), dtype=float32)</td>
      <td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 7, 7), dtype=float32)</td>
      <td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
    </tr>
    <tr>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 7, 7), dtype=float32)</td>
      <td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)</td>
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
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(36,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(72,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(18,), dtype=float32)</td>
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
      <td>Operand(type=Parameter, shape=(256,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(32, 1), dtype=float32)</td>
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
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(36,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(36, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(72,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(72, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(18,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(18, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
