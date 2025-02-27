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
      <td>Operand(type=Activation, shape=(1, 1000), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1000,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.embedder.embedder.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.0.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4654, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.0.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7654, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.0.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.0.shortcut.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.1.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.1.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.1.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.0.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.0.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.0.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.0.shortcut.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34654, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.1.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.1.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.1.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.2.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.2.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.2.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.3.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.3.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.3.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_61654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.4.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.4.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.4.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.5.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.5.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.5.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.0.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.0.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85654, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.0.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.0.shortcut.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.1.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_94654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.1.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_97654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.1.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.2.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.2.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_106654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.2.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_109654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.3.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.3.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_115654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.3.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_118654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>65</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.4.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_121654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>66</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.4.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_124654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>67</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.4.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_127654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>68</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.5.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_130654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>69</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.5.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_133654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>70</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.5.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_136654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>71</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.6.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_139654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>72</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.6.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_142654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>73</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.6.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_145654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>74</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.7.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_148654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>75</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.7.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_151654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>76</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.7.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_154654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>77</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.8.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_157654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>78</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.8.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_160654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>79</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.8.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_163654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>80</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.9.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_166654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>81</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.9.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_169654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>82</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.9.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_172654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>83</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.10.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_175654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>84</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.10.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_178654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>85</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.10.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_181654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>86</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.11.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_184654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>87</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.11.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_187654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>88</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.11.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_190654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>89</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.0.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_193654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>90</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.0.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_196654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>91</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>92</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.0.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_199654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>93</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.0.shortcut.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_202654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>94</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>95</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.1.layer.0.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_205654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>96</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.1.layer.1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_208654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>97</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 272, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(272, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>98</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.1.layer.3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_211654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>99</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
      <td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>100</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
      <td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>101</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
      <td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>102</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)</td>
      <td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>103</th>
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
      <th>104</th>
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
      <th>105</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>106</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>107</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>108</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 1, 1), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>109</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>110</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>111</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 8, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>112</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>114</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>115</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>116</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 128, 1, 1), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>117</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 64, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 3<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>118</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>119</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>120</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 64, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 3<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>121</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>122</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 48, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>123</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>124</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 192, 1, 1), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>125</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 64, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>126</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>127</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 48, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>128</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 64, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>129</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>130</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>131</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 512, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>132</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 512, 1, 1), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>133</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 64, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 17<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>134</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1088, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>135</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 128, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>136</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 1088, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>137</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 64, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 17<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>138</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(272, 1088, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>139</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 272, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 272, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>140</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1088), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1000), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_61605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_271605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_1141605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_2731605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1088,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_0454, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.embedder.embedder.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2654, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.0.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.0.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8654, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.0.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.0.shortcut.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.1.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.1.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.0.layers.1.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.0.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.0.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29654, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.0.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.0.shortcut.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.1.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.1.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.1.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44654, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.2.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>183</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.2.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>184</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.2.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>185</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.3.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>186</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.3.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>187</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.3.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_62654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>188</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.4.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_65654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>189</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.4.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_68654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>190</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.4.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_71654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>191</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.5.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_74654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>192</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.5.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>193</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.1.layers.5.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>194</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.0.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>195</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.0.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>196</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>197</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.0.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>198</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.0.shortcut.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>199</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.1.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_95654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>200</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.1.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>201</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.1.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>202</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.2.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>203</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.2.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_107654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>204</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.2.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>205</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.3.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>206</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.3.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>207</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.3.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>208</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.4.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_122654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>209</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.4.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>210</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.4.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>211</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.5.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>212</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.5.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_134654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>213</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.5.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_137654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>214</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.6.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_140654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>215</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.6.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_143654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>216</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.6.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_146654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>217</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.7.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_149654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>218</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.7.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_152654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>219</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.7.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_155654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>220</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.8.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_158654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>221</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.8.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_161654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>222</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.8.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_164654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>223</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.9.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_167654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>224</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.9.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_170654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>225</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.9.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_173654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>226</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.10.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_176654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>227</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.10.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_179654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>228</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.10.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_182654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>229</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.11.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_185654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>230</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.11.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_188654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>231</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.2.layers.11.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_191654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>232</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.0.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_194654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>233</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.0.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_197654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>234</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>235</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.0.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_200654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>236</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.0.shortcut.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_203654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>237</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.1.layer.0.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_206654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>238</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.1.layer.1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_209654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>239</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=regnet.encoder.stages.3.layers.1.layer.3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_212654, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>240</th>
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
      <th>241</th>
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
      <th>242</th>
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
      <th>243</th>
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
      <th>244</th>
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
      <th>245</th>
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
      <th>246</th>
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
      <th>247</th>
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
      <th>248</th>
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
      <th>249</th>
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
      <th>250</th>
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
      <th>251</th>
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
      <th>252</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>253</th>
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
      <th>254</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>255</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>256</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>257</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>258</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 272, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>259</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)</td>
      <td>shape : (1, 1088, 1, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>260</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 192, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>261</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>262</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>263</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>264</th>
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
      <th>265</th>
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
      <th>266</th>
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
      <th>267</th>
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
      <th>268</th>
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
      <th>269</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)</td>
      <td>dim : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>270</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 1088, 1), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>271</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1000, 1088), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>272</th>
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
      <th>273</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>274</th>
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
      <th>275</th>
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
      <th>276</th>
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
      <th>277</th>
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
      <th>278</th>
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
      <th>279</th>
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
      <th>280</th>
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
      <th>281</th>
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
      <th>282</th>
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
      <th>283</th>
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
      <th>284</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(8,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>285</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(8, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>286</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>287</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(48,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>288</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(48, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>289</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(272, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>290</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>291</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(1088,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>292</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(272,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
