<h1>Unique ops configuration and compiler support info</h1>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Operation Details</th>
      <th colspan="4" halign="left">Component Passing Check</th>
      <th>Issues</th>
    </tr>
    <tr>
      <th></th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>5</th>
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
      <th>6</th>
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
      <th>7</th>
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
      <th>8</th>
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
      <th>9</th>
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
      <th>10</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 16, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
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
      <th>14</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
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
      <th>17</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree1.tree1.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>18</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree1.tree1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree1.project.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree1.tree2.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>21</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree1.tree2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>22</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree1.root.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_97382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>23</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree2.tree1.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree2.tree1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>25</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree2.tree2.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_109382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>26</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree2.tree2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>27</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=level4.tree2.root.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_118382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>28</th>
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
      <th>29</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>31</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>32</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>33</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=model.1.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=model.2.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>35</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=model.4.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>36</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.base_layer.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>37</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level0.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>38</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level1.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>39</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level2.tree1.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>40</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level2.tree1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level2.project.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level2.tree2.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level2.tree2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>44</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level2.root.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>45</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.tree1.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>46</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.tree1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.project.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.tree2.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.tree2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.root.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.tree1.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.tree1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.tree2.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.tree2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.root.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>56</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level5.tree1.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>57</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level5.tree1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_115396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>58</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level5.project.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_118396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>59</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level5.tree2.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_121396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>60</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level5.tree2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_124396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>61</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=backbone.level5.root.bn.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_127396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>62</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 3, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>63</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 2, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>65</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>66</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>67</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>68</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>69</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>70</th>
      <td>Concatenate</td>
      <td>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)</td>
      <td>axis : -4</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>71</th>
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
      <th>72</th>
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
      <th>73</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 3, 7, 7), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>74</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>75</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 16, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>76</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>77</th>
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
      <th>78</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>79</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>80</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>81</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>82</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>83</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>84</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>85</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>86</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>87</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>88</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 896, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>89</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>90</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>91</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1280, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>92</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>93</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>94</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>95</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>96</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>97</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>98</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1792, 64, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>100</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>101</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 256, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>102</th>
      <td>Conv2dTranspose</td>
      <td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 1, 4, 4), dtype=float32)</td>
      <td>stride : 2<br>padding : 1<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: conv2d_transpose</td>
    </tr>
    <tr>
      <th>103</th>
      <td>Conv2dTranspose</td>
      <td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1, 4, 4), dtype=float32)</td>
      <td>stride : 2<br>padding : 1<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: conv2d_transpose</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Conv2dTranspose</td>
      <td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1, 4, 4), dtype=float32)</td>
      <td>stride : 2<br>padding : 1<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: conv2d_transpose</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>106</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
      <td>dim : -3<br>start : 256<br>stop : 512<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>107</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
      <td>dim : -3<br>start : 512<br>stop : 768<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>108</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
      <td>dim : -3<br>start : 768<br>stop : 1024<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>109</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
      <td>dim : -3<br>start : 1024<br>stop : 1280<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>110</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
      <td>dim : -3<br>start : 1280<br>stop : 1536<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>111</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
      <td>dim : -3<br>start : 1536<br>stop : 1792<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>112</th>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)</td>
      <td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
      <td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>114</th>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
      <td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>115</th>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
      <td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>116</th>
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
      <th>117</th>
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
      <th>118</th>
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
      <th>119</th>
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
      <th>120</th>
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
      <th>121</th>
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
      <th>122</th>
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
      <th>123</th>
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
      <th>124</th>
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
      <th>125</th>
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
      <th>126</th>
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
      <th>127</th>
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
      <th>128</th>
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
      <th>129</th>
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
      <th>130</th>
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
      <th>131</th>
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
      <th>132</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>133</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_01342, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>134</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>135</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 16, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>136</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>137</th>
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
      <th>138</th>
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
      <th>139</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>140</th>
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
      <th>141</th>
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
      <th>142</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree1.tree1.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>143</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree1.tree1.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>144</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree1.project.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>145</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree1.tree2.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>146</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree1.tree2.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>147</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree1.root.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>148</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree2.tree1.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>149</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree2.tree1.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>150</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree2.tree2.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>151</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree2.tree2.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>152</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=level4.tree2.root.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119382, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>153</th>
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
      <th>154</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=model.1.model.3.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>155</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=model.2.model.3.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>156</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=model.4.model.3.model.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26214, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>157</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.base_layer.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>158</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>159</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level1.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>160</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level2.tree1.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>161</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level2.tree1.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>162</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level2.project.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>163</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level2.tree2.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>164</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level2.tree2.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>165</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level2.root.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>166</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.tree1.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>167</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.tree1.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>168</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.project.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>169</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.tree2.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>170</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.tree2.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>171</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree1.root.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>172</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.tree1.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>173</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.tree1.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>174</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.tree2.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>175</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.tree2.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>176</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level3.tree2.root.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>177</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level5.tree1.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>178</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level5.tree1.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>179</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level5.project.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>180</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level5.tree2.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_122396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>181</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level5.tree2.bn2.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>182</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=backbone.level5.root.bn.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128396, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>183</th>
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
      <th>184</th>
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
      <th>185</th>
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
      <th>186</th>
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
      <th>187</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>188</th>
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
      <th>189</th>
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
      <th>190</th>
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
      <th>191</th>
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
      <th>192</th>
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
      <th>193</th>
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
      <th>194</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>195</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 16, 224, 224), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>196</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>197</th>
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
      <th>198</th>
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
      <th>199</th>
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
      <th>200</th>
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
      <th>201</th>
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
      <th>202</th>
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
      <th>203</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(16,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>204</th>
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
      <th>205</th>
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
      <th>206</th>
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
      <th>207</th>
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
      <th>208</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(16, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>209</th>
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
      <th>210</th>
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
      <th>211</th>
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
      <th>212</th>
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
      <th>213</th>
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
      <th>214</th>
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
      <th>215</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(16,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>216</th>
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
      <th>217</th>
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
      <th>218</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(24, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>219</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(24,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>220</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(3,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>221</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(3, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>222</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(2,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>223</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(2, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
