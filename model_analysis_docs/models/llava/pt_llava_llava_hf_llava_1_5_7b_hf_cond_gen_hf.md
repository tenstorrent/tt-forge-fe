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
      <td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 577, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 576, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 596, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_510, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 32, 596, 596), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_560, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>AdvIndex</td>
      <td>Operand(type=Activation, shape=(2359296,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2441216,), dtype=int32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td>[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Broadcast</td>
      <td>Operand(type=Activation, shape=(1, 596, 1), dtype=uint1)</td>
      <td>dim : -1<br>shape : 4096</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cast</td>
      <td>Operand(type=Activation, shape=(1, 596, 4096), dtype=uint1)</td>
      <td>dtype : torch.float32</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 0 - data type mismatch: expected UInt8, got Float32</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cast</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=float32)</td>
      <td>dtype : torch.bool</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int8, compiled_model.dtype=torch.uint8</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Cast</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=float32)</td>
      <td>dtype : torch.int32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 576, 1024), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 596, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 596, 64), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 596, 64), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 336, 336), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 3, 14, 14), dtype=float32)</td>
      <td>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td>[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Cosine</td>
      <td>Operand(type=Activation, shape=(1, 596, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td>CumSum</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Embedding</td>
      <td>Operand(type=Constant, name=model.vision_tower.vision_model.embeddings.position_ids, dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(577, 1024), dtype=float32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td>[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Embedding</td>
      <td>Operand(type=Activation, shape=(1, 596), dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32064, 4096), dtype=float32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td>[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Equal</td>
      <td>Operand(type=Activation, shape=(1, 596), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn elementwise binary] RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 576, 4096), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(16, 577, 577), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>25</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 596), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>26</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 1<br>stop : 577<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>27</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
      <td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>28</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
      <td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>29</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 577, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 577), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 577, 577), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 577, 64), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 577, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 576, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 576, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_520, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32, 596, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 128, 596), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(32, 596, 596), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 596, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 11008), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 596, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 4096), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 32064), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Max</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=int32)<br><div align='center'>X</div>Operand(type=Constant, name=const_480, dtype=int32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Min</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=int32)<br><div align='center'>X</div>Operand(type=Constant, name=const_490, dtype=int32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 577, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 577, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 577, 4096), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 596, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td>Multiply</td>
      <td>Operand(type=Parameter, shape=(4096,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 596, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_530, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 596), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_550, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 596, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 596, 11008), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>56</th>
      <td>Reciprocal</td>
      <td>Operand(type=Activation, shape=(1, 596, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>57</th>
      <td>ReduceAvg</td>
      <td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)</td>
      <td>dim : -1<br>keep_dim : True</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>58</th>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)</td>
      <td>repeats : 1<br>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>59</th>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)</td>
      <td>repeats : 1<br>dim : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>60</th>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
      <td>repeats : 1<br>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>61</th>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
      <td>repeats : 1<br>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>62</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)</td>
      <td>shape : (2441216,)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)</td>
      <td>shape : (596, 4096)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1024, 24, 24), dtype=float32)</td>
      <td>shape : (1, 1024, 576, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>65</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)</td>
      <td>shape : (577, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>66</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)</td>
      <td>shape : (1, 577, 16, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>67</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(577, 1024), dtype=float32)</td>
      <td>shape : (1, 577, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>68</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 16, 577, 64), dtype=float32)</td>
      <td>shape : (16, 577, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>69</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 577, 64), dtype=float32)</td>
      <td>shape : (1, 16, 577, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>70</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 577, 16, 64), dtype=float32)</td>
      <td>shape : (577, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>71</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 576, 4096), dtype=float32)</td>
      <td>shape : (2359296,)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 2441216), dtype=int32)</td>
      <td>shape : (2441216,)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>73</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=float32)</td>
      <td>shape : (2441216,)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>74</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=float32)</td>
      <td>shape : (1, 596, 4096)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(596, 4096), dtype=float32)</td>
      <td>shape : (1, 596, 32, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>76</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(596, 4096), dtype=float32)</td>
      <td>shape : (1, 596, 4096)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>77</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
      <td>shape : (32, 596, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>78</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(32, 596, 596), dtype=float32)</td>
      <td>shape : (1, 32, 596, 596)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>79</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 596), dtype=float32)</td>
      <td>shape : (32, 596, 596)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>80</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 32, 128, 596), dtype=float32)</td>
      <td>shape : (32, 128, 596)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>81</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(32, 596, 128), dtype=float32)</td>
      <td>shape : (1, 32, 596, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>82</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 596, 32, 128), dtype=float32)</td>
      <td>shape : (596, 4096)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>83</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(596, 11008), dtype=float32)</td>
      <td>shape : (1, 596, 11008)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>84</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 577, 4096), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>85</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 596, 11008), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>86</th>
      <td>Sine</td>
      <td>Operand(type=Activation, shape=(1, 596, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>87</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(16, 577, 577), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>88</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 596), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>89</th>
      <td>Sqrt</td>
      <td>Operand(type=Activation, shape=(1, 596, 1), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>90</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 1024, 576, 1), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>91</th>
      <td>Subtract</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_470, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>92</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1024, 1024), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>93</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(4096, 1024), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>94</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1024, 4096), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>95</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(4096, 4096), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>96</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(11008, 4096), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>97</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(4096, 11008), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>98</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 1024, 576), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>99</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 577, 16, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>100</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(16, 577, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>101</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(16, 64, 577), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>102</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 16, 577, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>103</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 596, 32, 128), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>104</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 64, 596), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>105</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(32, 596, 128), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>106</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>107</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>108</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(32, 128, 596), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>109</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(32064, 4096), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>110</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>111</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Constant, name=model.model.rotary_emb.inv_freq, dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>112</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1, 64), dtype=float32)</td>
      <td>dim : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1, 596), dtype=uint1)</td>
      <td>dim : -1</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 0 - data type mismatch: expected UInt8, got Float32</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1, 1024), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>115</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(1, 596, 128), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>116</th>
      <td>Where</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2441216,), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_500, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>117</th>
      <td>Where</td>
      <td>Operand(type=Activation, shape=(2441216,), dtype=uint1)<br><div align='center'>X</div>Operand(type=Activation, shape=(2441216,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2441216,), dtype=float32)</td>
      <td></td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 0 - data type mismatch: expected UInt8, got Float32</td>
    </tr>
  </tbody>
</table>
