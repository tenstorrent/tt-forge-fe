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
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 1000), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1000,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(261,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(261,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 512, 50176), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 50176, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 1, 512, 50176), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 8, 512, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 1, 1, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(261,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(261,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>18</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>21</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(8, 512, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 128, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>22</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(8, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 512, 128), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>23</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1024, 512), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1000), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 261), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(50176, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(261, 261), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 261, 50176), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 50176), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 512, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(261, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_500, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 512, 50176), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 8, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
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
      <th>35</th>
      <td>RepeatInterleave</td>
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
      <td>repeats : 1<br>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>36</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
      <td>shape : (1, 1, 1, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>37</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)</td>
      <td>shape : (1, 1, 1, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>38</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 512, 50176), dtype=float32)</td>
      <td>shape : (1, 1, 512, 50176)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>39</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 512, 50176), dtype=float32)</td>
      <td>shape : (1, 512, 50176)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>40</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
      <td>shape : (512, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
      <td>shape : (1, 512, 8, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
      <td>shape : (1, 512, 1, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(512, 1024), dtype=float32)</td>
      <td>shape : (1, 512, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>44</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 8, 512, 128), dtype=float32)</td>
      <td>shape : (8, 512, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>45</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(8, 512, 512), dtype=float32)</td>
      <td>shape : (1, 8, 512, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>46</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 8, 512, 512), dtype=float32)</td>
      <td>shape : (8, 512, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 8, 128, 512), dtype=float32)</td>
      <td>shape : (8, 128, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(8, 512, 128), dtype=float32)</td>
      <td>shape : (1, 8, 512, 128)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 512, 8, 128), dtype=float32)</td>
      <td>shape : (512, 1024)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 1, 512), dtype=float32)</td>
      <td>shape : (1, 1, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 1024, 512), dtype=float32)</td>
      <td>shape : (1, 1024, 512)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 1000), dtype=float32)</td>
      <td>shape : (1, 1000)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 512, 261), dtype=float32)</td>
      <td>shape : (1, 512, 1, 261)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 224, 224, 3), dtype=float32)</td>
      <td>shape : (1, 50176, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)</td>
      <td>shape : (50176, 261)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>56</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)</td>
      <td>shape : (1, 50176, 1, 261)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>57</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(50176, 261), dtype=float32)</td>
      <td>shape : (1, 50176, 261)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>58</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 261, 50176), dtype=float32)</td>
      <td>shape : (1, 261, 50176)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>59</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 1, 512, 50176), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 8, 512, 512), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>61</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 1, 1, 512), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>62</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 1, 1, 1024), dtype=float32)</td>
      <td>dim : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>63</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 512, 1, 1024), dtype=float32)</td>
      <td>dim : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 512, 1, 261), dtype=float32)</td>
      <td>dim : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>65</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 50176, 1, 261), dtype=float32)</td>
      <td>dim : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>66</th>
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
      <th>67</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1000, 1024), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>68</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 512, 8, 128), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>69</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(8, 512, 128), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>70</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 8, 512, 128), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>71</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 8, 512, 128), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>72</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(8, 128, 512), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>73</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>74</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 512, 1, 1024), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>75</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 1, 512, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1024, 512), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>77</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(261, 1024), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>79</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 224, 3, 224), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>80</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(261, 261), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 50176, 1, 261), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 1, 50176, 261), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 261, 50176), dtype=float32)</td>
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
      <td>Operand(type=Parameter, shape=(1024, 261), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>86</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(1, 1024), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>87</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(512, 1024), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
