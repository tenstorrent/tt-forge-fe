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
      <td>Operand(type=Activation, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(336,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(24,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(112,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(48,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(56,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(8,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1001), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1001,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.conv_stem.first_conv.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.conv_stem.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.conv_stem.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.0.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.0.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.0.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.1.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.1.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.1.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.2.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.2.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.2.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.3.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.3.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.3.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.4.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.4.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.4.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.5.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.5.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.5.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_61414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.6.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.6.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.6.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.7.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.7.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.7.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.8.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.8.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.8.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.9.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.9.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_94414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.9.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_97414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.10.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.10.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.10.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_106414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.11.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_109414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.11.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.11.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_115414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.12.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_118414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.12.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_121414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.12.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_124414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.13.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_127414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.13.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_130414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.13.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_133414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.14.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_136414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.14.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_139414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.14.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_142414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.15.expand_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_145414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.15.conv_3x3.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_148414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.15.reduce_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_151414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.conv_1x1.normalization.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_154414, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 8, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 8, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 8, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 24, 24), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 48, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 12, 12), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 96, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 24, 6, 6), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 32, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 32, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 6, 6), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 56, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(56, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 56, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 3, 3), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 112, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1280, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>89</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 1280, 3, 3), dtype=float32)</td>
      <td>kernel_size : [3, 3]<br>stride : [3, 3]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn.reshape] RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>91</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>92</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>93</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 48, 12, 12), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>94</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>95</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 96, 6, 6), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>96</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>97</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>98</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 192, 3, 3), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>99</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>100</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 1280, 3, 3), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>101</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 3, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][TTIR to TTNN Conv2dOpConversionPattern] tt_forge_signal_handler tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &) adaptor.getPaddingBottom() == adaptor.getPaddingTop() TTNN only supports padding height/width attributes. Thus, padding_top must equal padding_bottom for the op to execute as expected</td>
    </tr>
    <tr>
      <th>102</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>103</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 16, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [16, 1, 1, 1], got [1, 8, 128, 128]</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 8, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 8, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [8, 1, 1, 1], got [1, 48, 384, 384]</td>
    </tr>
    <tr>
      <th>105</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][TTIR to TTNN Conv2dOpConversionPattern] tt_forge_signal_handler tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &) adaptor.getPaddingBottom() == adaptor.getPaddingTop() TTNN only supports padding height/width attributes. Thus, padding_top must equal padding_bottom for the op to execute as expected</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 48, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 8, 384, 384]</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 8, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 8, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [8, 1, 1, 1], got [1, 48, 384, 384]</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>109</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][TTIR to TTNN Conv2dOpConversionPattern] tt_forge_signal_handler tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &) adaptor.getPaddingBottom() == adaptor.getPaddingTop() TTNN only supports padding height/width attributes. Thus, padding_top must equal padding_bottom for the op to execute as expected</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 48, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 16, 768, 768]</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 16, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [16, 1, 1, 1], got [1, 96, 1536, 1536]</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][TTIR to TTNN Conv2dOpConversionPattern] tt_forge_signal_handler tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &) adaptor.getPaddingBottom() == adaptor.getPaddingTop() TTNN only supports padding height/width attributes. Thus, padding_top must equal padding_bottom for the op to execute as expected</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 96, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 16, 1536, 1536]</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 24, 2304, 2304]</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 24, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 144, 3456, 3456]</td>
    </tr>
    <tr>
      <th>117</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>118</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 24, 3456, 3456]</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 144, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 32, 4608, 4608]</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [32, 1, 1, 1], got [1, 192, 6144, 6144]</td>
    </tr>
    <tr>
      <th>121</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>122</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[MLIR][TTIR to TTNN Conv2dOpConversionPattern] tt_forge_signal_handler tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &) adaptor.getPaddingBottom() == adaptor.getPaddingTop() TTNN only supports padding height/width attributes. Thus, padding_top must equal padding_bottom for the op to execute as expected</td>
    </tr>
    <tr>
      <th>123</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 32, 6144, 6144]</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 192, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 56, 10752, 10752]</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 56, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 56, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [56, 1, 1, 1], got [1, 336, 18816, 18816]</td>
    </tr>
    <tr>
      <th>126</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>127</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 336, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [336, 1, 1, 1], got [1, 56, 18816, 18816]</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 336, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [336, 1, 1, 1], got [1, 112, 37632, 37632]</td>
    </tr>
    <tr>
      <th>129</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 112, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 112, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [112, 1, 1, 1], got [1, 1280, 143360, 143360]</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>131</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1001), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_9965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_27965, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_3091605, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_91285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(144,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_331285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(336,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(336,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_01342, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_61342, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_15454, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(24,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(24,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_78454, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(112,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(112,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_0894, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(48,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(48,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_57894, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(56,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(56,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(56,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(56,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_3680, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(8,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(8,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.conv_stem.first_conv.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.conv_stem.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.conv_stem.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.0.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.0.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.0.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.1.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.1.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.1.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.2.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.2.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.2.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.3.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.3.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.3.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.4.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.4.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.4.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.5.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.5.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.5.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_62414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.6.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_65414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.6.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_68414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.6.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_71414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.7.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_74414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.7.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.7.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.8.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.8.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.8.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.9.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.9.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_95414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.9.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.10.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.10.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.10.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_107414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.11.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.11.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.11.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.12.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.12.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_122414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.12.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.13.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.13.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.13.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_134414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.14.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_137414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.14.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_140414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.14.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_143414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.15.expand_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_146414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.15.conv_3x3.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_149414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.layer.15.reduce_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_152414, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=mobilenet_v2.conv_1x1.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_155414, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 8, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 8, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 48, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 96, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 32, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 56, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(56, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 112, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1280, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>238</th>
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
      <th>239</th>
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
      <th>240</th>
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
      <th>241</th>
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
      <th>242</th>
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
      <th>243</th>
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
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(24,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>246</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(112,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>247</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(48,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>248</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(56,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>249</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(8,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>250</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(96, 1, 3, 3), dtype=float32)</td>
      <td>shape : (96, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>251</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(144, 1, 3, 3), dtype=float32)</td>
      <td>shape : (144, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>252</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)</td>
      <td>shape : (1, 1280, 1, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>253</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(48, 1, 3, 3), dtype=float32)</td>
      <td>shape : (48, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>254</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(192, 1, 3, 3), dtype=float32)</td>
      <td>shape : (192, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>255</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(336, 1, 3, 3), dtype=float32)</td>
      <td>shape : (336, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>256</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(16, 1, 3, 3), dtype=float32)</td>
      <td>shape : (16, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>257</th>
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
      <th>258</th>
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
      <th>259</th>
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
      <th>260</th>
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
      <th>261</th>
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
      <th>262</th>
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
      <th>263</th>
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
      <th>264</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(24,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(112,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(48,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(56,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(8,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1280, 1), dtype=float32)</td>
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
      <td>Operand(type=Parameter, shape=(1001, 1280), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(96, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1280, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(144,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(144, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(336,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(336, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(8, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(24,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(24, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(112,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(112, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(48, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(48,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>293</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(56,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>294</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(56, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>295</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(8,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
