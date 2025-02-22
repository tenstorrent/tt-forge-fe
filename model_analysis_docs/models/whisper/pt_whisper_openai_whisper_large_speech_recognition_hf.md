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
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1280, 3000), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1280, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1500, 1280), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1500, 5120), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(5120,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 5120), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(5120,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>AdvIndex</td>
      <td>Operand(type=Parameter, shape=(448, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 80, 3, 1), dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1280, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280, 3, 1), dtype=float32)</td>
      <td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Embedding</td>
      <td>Operand(type=Activation, shape=(1, 1), dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(51865, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 1280, 3000), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 1280, 1500), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 1500, 5120), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 1, 5120), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>18</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>21</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 20, 1500, 1500), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>22</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 1500, 5120), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>23</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 1500), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 1, 5120), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>25</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>26</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>27</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>28</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 64, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>29</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(20, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>31</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>32</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 64, 1500), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>33</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(20, 1500, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 5120), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>35</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1500, 5120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5120, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>36</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 64, 1500), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>37</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(20, 1, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>38</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 5120), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>39</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1, 5120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5120, 1280), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>40</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 51865), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1), dtype=int64)</td>
      <td>shape : (1, 1)</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 80, 3000), dtype=float32)</td>
      <td>shape : (1, 80, 3000, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>45</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)</td>
      <td>shape : (1, 1280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>46</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)</td>
      <td>shape : (1, 1, 20, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1280), dtype=float32)</td>
      <td>shape : (1, 1, 1280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1280), dtype=float32)</td>
      <td>shape : (1, 1, 20, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 64), dtype=float32)</td>
      <td>shape : (20, 1, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(20, 1, 1), dtype=float32)</td>
      <td>shape : (1, 20, 1, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)</td>
      <td>shape : (20, 1, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 20, 64, 1), dtype=float32)</td>
      <td>shape : (20, 64, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)</td>
      <td>shape : (1, 20, 1, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 20, 64), dtype=float32)</td>
      <td>shape : (1, 1280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(1280, 80, 3), dtype=float32)</td>
      <td>shape : (1280, 80, 3, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>56</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1280, 3000, 1), dtype=float32)</td>
      <td>shape : (1, 1280, 3000)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>57</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1280, 3000), dtype=float32)</td>
      <td>shape : (1, 1280, 3000, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>58</th>
      <td>Reshape</td>
      <td>Operand(type=Parameter, shape=(1280, 1280, 3), dtype=float32)</td>
      <td>shape : (1280, 1280, 3, 1)</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td>[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1280, 1500, 1), dtype=float32)</td>
      <td>shape : (1, 1280, 1500)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>60</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)</td>
      <td>shape : (1500, 1280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>61</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)</td>
      <td>shape : (1, 1500, 20, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>62</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1500, 1280), dtype=float32)</td>
      <td>shape : (1, 1500, 1280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>63</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1500, 1280), dtype=float32)</td>
      <td>shape : (1, 1500, 20, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 20, 1500, 64), dtype=float32)</td>
      <td>shape : (20, 1500, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>65</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(20, 1500, 1500), dtype=float32)</td>
      <td>shape : (1, 20, 1500, 1500)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>66</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 20, 1500, 1500), dtype=float32)</td>
      <td>shape : (20, 1500, 1500)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>67</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 20, 64, 1500), dtype=float32)</td>
      <td>shape : (20, 64, 1500)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>68</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)</td>
      <td>shape : (1, 20, 1500, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>69</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1500, 20, 64), dtype=float32)</td>
      <td>shape : (1500, 1280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>70</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(20, 1, 1500), dtype=float32)</td>
      <td>shape : (1, 20, 1, 1500)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>71</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 1500), dtype=float32)</td>
      <td>shape : (20, 1, 1500)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>72</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>73</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 20, 1500, 1500), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>74</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 1500), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>75</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1280, 1280), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>76</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 1, 20, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>77</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>78</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>79</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 20, 1, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>80</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(20, 64, 1), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>81</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 1280, 1500), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>82</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 1500, 20, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>83</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>84</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 20, 1500, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>85</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 20, 1500, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>86</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(20, 64, 1500), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>87</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(5120, 1280), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>88</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1280, 5120), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>89</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(51865, 1280), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>90</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
