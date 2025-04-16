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
      <td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60, dtype=float32)</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>&#xFFFD;</td>
      <td>[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Add</td>
      <td>Operand(type=Activation, shape=(1, 256, 51200), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(51200,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>Embedding</td>
      <td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(51200, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)</td>
      <td>approximate : "tanh"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 768<br>stop : 1024<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 1536<br>stop : 1792<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 2304<br>stop : 2560<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 512<br>stop : 768<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>18</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 1280<br>stop : 1536<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 2048<br>stop : 2304<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 2816<br>stop : 3072<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>21</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 256<br>stop : 512<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>22</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 1024<br>stop : 1280<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>23</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 1792<br>stop : 2048<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td>Index</td>
      <td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
      <td>dim : -2<br>start : 2560<br>stop : 2816<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>25</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
      <td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>26</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
      <td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>27</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
      <td>dim : -1<br>start : 1<br>stop : 32<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>28</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
      <td>dim : -1<br>start : 0<br>stop : 32<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>29</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 256), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 256, 64), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 51200), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>36</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>37</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>38</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>39</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>40</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)</td>
      <td>shape : (256, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(256, 1024), dtype=float32)</td>
      <td>shape : (1, 256, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(256, 1024), dtype=float32)</td>
      <td>shape : (1, 256, 4, 256)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 16, 256, 64), dtype=float32)</td>
      <td>shape : (16, 256, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>44</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 256, 256), dtype=float32)</td>
      <td>shape : (1, 16, 256, 256)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>45</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)</td>
      <td>shape : (16, 256, 256)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>46</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 256, 64), dtype=float32)</td>
      <td>shape : (1, 16, 256, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
      <td>shape : (256, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 256, 4, 256), dtype=float32)</td>
      <td>shape : (1, 256, 16, 64)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 16, 2), dtype=float32)</td>
      <td>shape : (1, 256, 16, 32, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 16, 64, 256), dtype=float32)</td>
      <td>shape : (16, 64, 256)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(256, 4096), dtype=float32)</td>
      <td>shape : (1, 256, 4096)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 32, 1), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td>Stack</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
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
      <th>56</th>
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
      <th>57</th>
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
      <th>58</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>59</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(16, 256, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>60</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(16, 64, 256), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>61</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 16, 256, 64), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>62</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 16, 256, 64), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>63</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(51200, 1024), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
