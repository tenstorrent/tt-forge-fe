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
      <td>Operand(type=Activation, shape=(240,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(336,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(480,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(672,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(960,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1454, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(24,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(40,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(80,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(112,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(48,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(56,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(120,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120,), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.0.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(8,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.0.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.0.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.0.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.1.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.1.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.1.0.bn_dw.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.1.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(12,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 12, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.1.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.1.0.shortcut.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.1.0.shortcut.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(36,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.2.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.2.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.3.0.bn_dw.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(72,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_57680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(20,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 20, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.3.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_63680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.3.0.shortcut.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_66680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.3.0.shortcut.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_69680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_72680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(60,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_75680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.4.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.5.0.bn_dw.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.5.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_95680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.5.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.5.0.shortcut.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.5.0.shortcut.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_107680, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(100,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(100,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(100, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.6.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.6.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116680, dtype=float32)</td>
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
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>99</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(92,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(92,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>100</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(92, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>101</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_122680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>102</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>103</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>104</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>105</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_134680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>106</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_137680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>107</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_140680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>108</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_143680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>109</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_146680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>110</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_148680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_151680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>112</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 56, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(56, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_154680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>114</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.shortcut.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_157680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>115</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.shortcut.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_160680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>116</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_163680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>117</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_166680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>118</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 168, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(168, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>119</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_168680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_171680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>121</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_174680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>122</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_177680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>123</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_180680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>124</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.bn_dw.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_183680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>125</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_188680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>126</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>127</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_191680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>128</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.shortcut.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_194680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>129</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>130</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.shortcut.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_197680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>131</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>132</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>133</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_200680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>134</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>135</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_203680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>136</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_206680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>137</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_209680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>138</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_212680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>139</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_215680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>140</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_217680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_220680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>142</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_223680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>143</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_226680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>144</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_229680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>145</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_232680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>146</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_235680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>147</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_238680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>148</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_241680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>149</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost2.primary_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_246680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>150</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost2.cheap_operation.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_249680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>151</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.9.0.bn1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_252680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>152</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>153</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_431016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>154</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_461016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>155</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_491016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>156</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_641016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>157</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_671016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>158</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_701016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>159</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_961016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>160</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>161</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_991016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>162</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1021016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>163</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1191016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>164</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1221016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>165</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1251016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>166</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1491016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>167</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(200,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>168</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>169</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1521016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>170</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1551016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>171</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1701016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>172</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(184,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>173</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>174</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1731016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>175</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1761016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>176</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1911016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>177</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1941016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>178</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1971016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>179</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2121016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>180</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2151016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>181</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2181016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>182</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2411016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>183</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2441016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>184</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2471016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>185</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2641016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>186</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2671016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>187</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2701016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>188</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2961016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>189</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>190</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2991016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>191</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3021016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>192</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3171016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>193</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3201016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>194</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3231016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>195</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3401016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>196</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3431016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>197</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3461016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>198</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.short_conv.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3611016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>199</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.short_conv.3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3641016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>200</th>
      <td>Add</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.short_conv.5.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3671016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>201</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)</td>
      <td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>202</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>203</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>204</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>205</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>206</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)</td>
      <td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>207</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>208</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>209</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>210</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>211</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)</td>
      <td>min : 0.0<br>max : 6.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>212</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>213</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>214</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 12, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>215</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>216</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 20, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 20, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>217</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>218</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>219</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>220</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>221</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>222</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>223</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 56, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>224</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>225</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>226</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>227</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>228</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>229</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>230</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>231</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 3, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>232</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 16, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>233</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>234</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 16, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>235</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12, 48, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>236</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 12, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 12<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>237</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>238</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 16, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>239</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 24, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>240</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 36<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>241</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12, 72, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>242</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 5, 5), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 72<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>243</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20, 72, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>244</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 20, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>245</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20, 72, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>246</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 20, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 20<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>247</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 5, 5), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>248</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 24, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 24, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>249</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 40, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>250</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 60<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>251</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 120, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>252</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 32, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>253</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20, 120, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>254</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 40, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>255</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 120<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>256</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 240, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>257</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 40<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>258</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 3, 3), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 40<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>259</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 40, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>260</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(100, 80, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>261</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(100, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 100<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>262</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 200, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>263</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(92, 80, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>264</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(92, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 92<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>265</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 184, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>266</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 80, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>267</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>268</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 480, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>269</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 120, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>270</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 480, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>271</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 56, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(56, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 56<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>272</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 80<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>273</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 80, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>274</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 112, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>275</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>276</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(168, 672, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>277</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 168, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 168, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>278</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 672, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>279</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 672, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>280</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 80<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>281</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 5, 5), dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 112<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>282</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 112, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>283</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 160, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>284</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 3, 3), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>285</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 960, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>286</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 960, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>287</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 240, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>288</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>289</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 960, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>290</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 24, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 24, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>291</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1, 5), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 72<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>292</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 5, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 72<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>293</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 40, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>294</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 1, 5), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 120<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>295</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 5, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 120<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>296</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 40, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>297</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1, 5), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>298</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 5, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>299</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(200, 80, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>300</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200, 1, 1, 5), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 200<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>301</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200, 1, 5, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 200<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>302</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(184, 80, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>303</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184, 1, 1, 5), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 184<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>304</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184, 1, 5, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 184<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>305</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 80, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>306</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1, 5), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>307</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 5, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>308</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 112, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>309</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1, 5), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>310</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>311</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 160, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>312</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1, 5), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>313</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>314</th>
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
      <th>315</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1000), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>316</th>
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
      <th>317</th>
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
      <th>318</th>
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
      <th>319</th>
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
      <th>320</th>
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
      <th>321</th>
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
      <th>322</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_781285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>323</th>
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
      <th>324</th>
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
      <th>325</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1021285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>326</th>
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
      <th>327</th>
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
      <th>328</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1891285, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>329</th>
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
      <th>330</th>
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
      <th>331</th>
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
      <th>332</th>
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
      <th>333</th>
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
      <th>334</th>
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
      <th>335</th>
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
      <th>336</th>
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
      <th>337</th>
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
      <th>338</th>
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
      <th>339</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2454, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>340</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>341</th>
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
      <th>342</th>
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
      <th>343</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>344</th>
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
      <th>345</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_33454, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>346</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(40,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>347</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>348</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(40,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>349</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>350</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>351</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_51454, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>352</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(80,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>353</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>354</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(80,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>355</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>356</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>357</th>
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
      <th>358</th>
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
      <th>359</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>360</th>
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
      <th>361</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>362</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>363</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>364</th>
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
      <th>365</th>
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
      <th>366</th>
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
      <th>367</th>
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
      <th>368</th>
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
      <th>369</th>
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
      <th>370</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>371</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>372</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>373</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_105510, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>374</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(120,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>375</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(120,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>376</th>
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
      <th>377</th>
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
      <th>378</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>379</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.0.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>380</th>
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
      <th>381</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.0.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>382</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.0.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>383</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.0.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>384</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.1.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>385</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.1.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>386</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>387</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.1.0.bn_dw.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>388</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_24680, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>389</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(12,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>390</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 12, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>391</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.1.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>392</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(12,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>393</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.1.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>394</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>395</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.1.0.shortcut.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>396</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.1.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>397</th>
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
      <th>398</th>
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
      <th>399</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>400</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>401</th>
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
      <th>402</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>403</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>404</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>405</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>406</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>407</th>
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
      <th>408</th>
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
      <th>409</th>
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
      <th>410</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.bn_dw.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>411</th>
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
      <th>412</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>413</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>414</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_59680, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>415</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(20,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>416</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 20, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>417</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_61680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>418</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(20,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>419</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>420</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 24, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>421</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.shortcut.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>422</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>423</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_71680, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>424</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(60,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>425</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>426</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>427</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(60,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>428</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>429</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_78680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>430</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>431</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_81680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>432</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_84680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>433</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>434</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_87680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>435</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_90680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>436</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.bn_dw.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_93680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>437</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>438</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_96680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>439</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_99680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>440</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.shortcut.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_102680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>441</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_105680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>442</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_106680, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(100,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>443</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(100,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(100,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>444</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(100, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>445</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_108680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>446</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(100,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(100,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>447</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_111680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>448</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_114680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>449</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_117680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>450</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_118680, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(92,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>451</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(92,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(92,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>452</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(92, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>453</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_120680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>454</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(92,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(92,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>455</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_123680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>456</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_126680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>457</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_129680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>458</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_132680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>459</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_135680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>460</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_138680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>461</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_141680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>462</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_144680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>463</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_147680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>464</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_149680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>465</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 56, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(56, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>466</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_152680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>467</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_155680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>468</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.shortcut.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_158680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>469</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_161680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>470</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_164680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>471</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_167680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>472</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_169680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>473</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_172680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>474</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_175680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>475</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_178680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>476</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_181680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>477</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.bn_dw.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_184680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>478</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>479</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_189680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>480</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_192680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>481</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>482</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.shortcut.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_195680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>483</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>484</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_198680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>485</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>486</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_201680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>487</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_204680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>488</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_207680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>489</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_210680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>490</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_213680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>491</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_216680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>492</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_218680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>493</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>494</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_221680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>495</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_224680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>496</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_227680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>497</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_230680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>498</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_233680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>499</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_236680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>500</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_239680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>501</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_242680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>502</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_247680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>503</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost2.cheap_operation.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_250680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>504</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.9.0.bn1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_253680, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>505</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_441016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>506</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_471016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>507</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.2.0.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_501016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>508</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>509</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_651016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>510</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_681016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>511</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.3.0.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_711016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>512</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>513</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_971016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>514</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1001016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>515</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.4.0.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1031016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>516</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>517</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1201016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>518</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1231016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>519</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.5.0.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1261016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>520</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1481016, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>521</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(200,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>522</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>523</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1501016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>524</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(200,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>525</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1531016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>526</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.0.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1561016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>527</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 200, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>528</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_1691016, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>529</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(184,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>530</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>531</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1711016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>532</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(184,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>533</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1741016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>534</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.1.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1771016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>535</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 184, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>536</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1921016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>537</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1951016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>538</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.2.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1981016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>539</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2131016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>540</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2161016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>541</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.3.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2191016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>542</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2421016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>543</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2451016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>544</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.6.4.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2481016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>545</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2651016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>546</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2681016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>547</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.7.0.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2711016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>548</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>549</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2971016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>550</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3001016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>551</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.0.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3031016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>552</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3181016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>553</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3211016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>554</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.1.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3241016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>555</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3411016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>556</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3441016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>557</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.2.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3471016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>558</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.short_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3621016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>559</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.short_conv.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3651016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>560</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=blocks.8.3.ghost1.short_conv.5.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3681016, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>561</th>
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
      <th>562</th>
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
      <th>563</th>
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
      <th>564</th>
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
      <th>565</th>
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
      <th>566</th>
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
      <th>567</th>
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
      <th>568</th>
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
      <th>569</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(40,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>570</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(80,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>571</th>
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
      <th>572</th>
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
      <th>573</th>
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
      <th>574</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>575</th>
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
      <th>576</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(12,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>577</th>
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
      <th>578</th>
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
      <th>579</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(20,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>580</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(60,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>581</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(100,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>582</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(92,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>583</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>584</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>585</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)</td>
      <td>dim : -2<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>586</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 480, 1, 14), dtype=float32)</td>
      <td>dim : -1<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>587</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)</td>
      <td>dim : -2<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>588</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 672, 1, 14), dtype=float32)</td>
      <td>dim : -1<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>589</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)</td>
      <td>dim : -2<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>590</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 672, 1, 7), dtype=float32)</td>
      <td>dim : -1<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>591</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td>dim : -2<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>592</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 72, 1, 28), dtype=float32)</td>
      <td>dim : -1<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>593</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
      <td>dim : -2<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>594</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 120, 1, 28), dtype=float32)</td>
      <td>dim : -1<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>595</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)</td>
      <td>dim : -2<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>596</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 960, 1, 7), dtype=float32)</td>
      <td>dim : -1<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>597</th>
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
      <th>598</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>599</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>600</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>601</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>602</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>603</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>604</th>
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
      <th>605</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>606</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>607</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>608</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>609</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>610</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>611</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 168, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>612</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>613</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>614</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>615</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(240, 1, 3, 3), dtype=float32)</td>
      <td>shape : (240, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>616</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(480, 1, 3, 3), dtype=float32)</td>
      <td>shape : (480, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>617</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(672, 1, 5, 5), dtype=float32)</td>
      <td>shape : (672, 1, 5, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>618</th>
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
      <th>619</th>
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
      <th>620</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(24, 1, 3, 3), dtype=float32)</td>
      <td>shape : (24, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>621</th>
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
      <th>622</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(8, 1, 3, 3), dtype=float32)</td>
      <td>shape : (8, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>623</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(12, 1, 3, 3), dtype=float32)</td>
      <td>shape : (12, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>624</th>
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
      <th>625</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(36, 1, 3, 3), dtype=float32)</td>
      <td>shape : (36, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>626</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(72, 1, 5, 5), dtype=float32)</td>
      <td>shape : (72, 1, 5, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>627</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(20, 1, 3, 3), dtype=float32)</td>
      <td>shape : (20, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>628</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(24, 1, 5, 5), dtype=float32)</td>
      <td>shape : (24, 1, 5, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>629</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(60, 1, 3, 3), dtype=float32)</td>
      <td>shape : (60, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>630</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(120, 1, 3, 3), dtype=float32)</td>
      <td>shape : (120, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>631</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(40, 1, 3, 3), dtype=float32)</td>
      <td>shape : (40, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>632</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(100, 1, 3, 3), dtype=float32)</td>
      <td>shape : (100, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>633</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(92, 1, 3, 3), dtype=float32)</td>
      <td>shape : (92, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>634</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(56, 1, 3, 3), dtype=float32)</td>
      <td>shape : (56, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>635</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(80, 1, 3, 3), dtype=float32)</td>
      <td>shape : (80, 1, 3, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>636</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(112, 1, 5, 5), dtype=float32)</td>
      <td>shape : (112, 1, 5, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>637</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(72, 1, 1, 5), dtype=float32)</td>
      <td>shape : (72, 1, 1, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>638</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(72, 1, 5, 1), dtype=float32)</td>
      <td>shape : (72, 1, 5, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>639</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(120, 1, 1, 5), dtype=float32)</td>
      <td>shape : (120, 1, 1, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>640</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(120, 1, 5, 1), dtype=float32)</td>
      <td>shape : (120, 1, 5, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>641</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(240, 1, 1, 5), dtype=float32)</td>
      <td>shape : (240, 1, 1, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>642</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(240, 1, 5, 1), dtype=float32)</td>
      <td>shape : (240, 1, 5, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>643</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(200, 1, 1, 5), dtype=float32)</td>
      <td>shape : (200, 1, 1, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>644</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(200, 1, 5, 1), dtype=float32)</td>
      <td>shape : (200, 1, 5, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>645</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(184, 1, 1, 5), dtype=float32)</td>
      <td>shape : (184, 1, 1, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>646</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(184, 1, 5, 1), dtype=float32)</td>
      <td>shape : (184, 1, 5, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>647</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(480, 1, 1, 5), dtype=float32)</td>
      <td>shape : (480, 1, 1, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>648</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(480, 1, 5, 1), dtype=float32)</td>
      <td>shape : (480, 1, 5, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>649</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(672, 1, 1, 5), dtype=float32)</td>
      <td>shape : (672, 1, 1, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>650</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(672, 1, 5, 1), dtype=float32)</td>
      <td>shape : (672, 1, 5, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>651</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(960, 1, 1, 5), dtype=float32)</td>
      <td>shape : (960, 1, 1, 5)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>652</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(960, 1, 5, 1), dtype=float32)</td>
      <td>shape : (960, 1, 5, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>653</th>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>654</th>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)</td>
      <td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>655</th>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
      <td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>656</th>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)</td>
      <td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>657</th>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)</td>
      <td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>658</th>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)</td>
      <td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>659</th>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)</td>
      <td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>660</th>
      <td>Resize2d</td>
      <td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)</td>
      <td>sizes : [7, 7]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td>[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
    </tr>
    <tr>
      <th>661</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>662</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>663</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>664</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>665</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>666</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>667</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>668</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>669</th>
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
      <th>670</th>
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
      <th>671</th>
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
      <th>672</th>
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
      <th>673</th>
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
      <th>674</th>
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
      <th>675</th>
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
      <th>676</th>
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
      <th>677</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(40,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>678</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(80,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>679</th>
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
      <th>680</th>
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
      <th>681</th>
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
      <th>682</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>683</th>
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
      <th>684</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(12,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>685</th>
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
      <th>686</th>
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
      <th>687</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(20,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>688</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(60,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>689</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(100,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>690</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(92,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>691</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>692</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(184,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>693</th>
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
      <th>694</th>
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
      <th>695</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1000, 1280), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>696</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>697</th>
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
      <th>698</th>
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
      <th>699</th>
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
      <th>700</th>
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
      <th>701</th>
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
      <th>702</th>
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
      <th>703</th>
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
      <th>704</th>
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
      <th>705</th>
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
      <th>706</th>
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
      <th>707</th>
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
      <th>708</th>
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
      <th>709</th>
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
      <th>710</th>
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
      <th>711</th>
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
      <th>712</th>
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
      <th>713</th>
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
      <th>714</th>
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
      <th>715</th>
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
      <th>716</th>
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
      <th>717</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(40,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>718</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(40, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>719</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(240,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>720</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(80,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>721</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(80, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>722</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(20,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>723</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(20, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>724</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(480,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>725</th>
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
      <th>726</th>
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
      <th>727</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(672,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>728</th>
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
      <th>729</th>
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
      <th>730</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(12, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>731</th>
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
      <th>732</th>
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
      <th>733</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(960,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>734</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(120,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>735</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(120, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>736</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(8,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>737</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(12,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>738</th>
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
      <th>739</th>
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
      <th>740</th>
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
      <th>741</th>
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
      <th>742</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(72,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>743</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(20,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>744</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(60,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>745</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(60, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>746</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(120,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>747</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(100,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>748</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(100, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>749</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(92,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>750</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(92, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>751</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(168,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>752</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(168, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>753</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(200,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>754</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(200, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>755</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(184,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>756</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(184, 1), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
