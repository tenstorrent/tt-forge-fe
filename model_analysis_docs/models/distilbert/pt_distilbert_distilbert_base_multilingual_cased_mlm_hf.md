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
      <td>Abs</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3072,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 119547), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(119547,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>Broadcast</td>
      <td>Operand(type=Activation, shape=(1, 1, 1, 128), dtype=uint1)</td>
      <td>dim : -3<br>shape : 12</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Broadcast</td>
      <td>Operand(type=Activation, shape=(1, 1, 1, 128), dtype=uint1)</td>
      <td>dim : -4<br>shape : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>Broadcast</td>
      <td>Operand(type=Activation, shape=(1, 12, 1, 128), dtype=uint1)</td>
      <td>dim : -2<br>shape : 128</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cast</td>
      <td>Operand(type=Activation, shape=(1, 128), dtype=int64)</td>
      <td>dtype : torch.bool</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cast</td>
      <td>Operand(type=Activation, shape=(1, 128), dtype=uint1)</td>
      <td>dtype : torch.int32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cast</td>
      <td>Operand(type=Activation, shape=(1, 128), dtype=int32)</td>
      <td>dtype : torch.bool</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Cast</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=uint1)</td>
      <td>dtype : torch.float32</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>Clip</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
      <td>min : 0.0<br>max : 1.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>Embedding</td>
      <td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>Embedding</td>
      <td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(119547, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 128, 3072), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>18</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td>Greater</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3115, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn elementwise binary] RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>22</th>
      <td>Index</td>
      <td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
      <td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>23</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 0.0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>25</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(12, 128, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>26</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(12, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 128, 64), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>27</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 119547), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>31</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_0115, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>32</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>33</th>
      <td>Multiply</td>
      <td>Operand(type=Constant, name=const_4115, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
      <td>shape : (128, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>35</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
      <td>shape : (1, 128, 12, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>36</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(128, 768), dtype=float32)</td>
      <td>shape : (1, 128, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>37</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 64), dtype=float32)</td>
      <td>shape : (12, 128, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>38</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(12, 128, 128), dtype=float32)</td>
      <td>shape : (1, 12, 128, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>39</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
      <td>shape : (12, 128, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>40</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 12, 64, 128), dtype=float32)</td>
      <td>shape : (12, 64, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(12, 128, 64), dtype=float32)</td>
      <td>shape : (1, 12, 128, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 128, 12, 64), dtype=float32)</td>
      <td>shape : (128, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 128), dtype=uint1)</td>
      <td>shape : (1, 1, 1, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>44</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>45</th>
      <td>Subtract</td>
      <td>Operand(type=Constant, name=const_2115, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128), dtype=int32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>46</th>
      <td>Subtract</td>
      <td>Operand(type=Constant, name=const_1115, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(768, 768), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(3072, 768), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(768, 3072), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 128, 12, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(12, 128, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 12, 128, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(12, 64, 128), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(119547, 768), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
