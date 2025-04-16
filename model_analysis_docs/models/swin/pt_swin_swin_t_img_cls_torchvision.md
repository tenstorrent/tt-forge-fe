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
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(64, 49, 288), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(64, 3, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 49, 49), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(64, 49, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 56, 56, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 3, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 49, 576), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 6, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 6, 49, 49), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 49, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 28, 28, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 16, 6, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(4, 49, 1152), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1152,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(4, 12, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 49, 49), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(4, 49, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 14, 14, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 4, 12, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 49, 2304), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 24, 49, 49), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 7, 7, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3072,), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td>AdvIndex</td>
      <td>Operand(type=Parameter, shape=(169, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.1.0.attn.relative_position_index, dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>31</th>
      <td>AdvIndex</td>
      <td>Operand(type=Parameter, shape=(169, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.3.0.attn.relative_position_index, dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>32</th>
      <td>AdvIndex</td>
      <td>Operand(type=Parameter, shape=(169, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.5.0.attn.relative_position_index, dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>33</th>
      <td>AdvIndex</td>
      <td>Operand(type=Parameter, shape=(169, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.7.0.attn.relative_position_index, dtype=int64)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td>AvgPool2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)</td>
      <td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>35</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 53, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 56, 96), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>36</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 56, 53, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 3, 96), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>37</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 3, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 53, 56, 96), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>38</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 56, 3, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 53, 96), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>39</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 96), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>40</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 25, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 28, 192), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 28, 25, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 3, 192), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 3, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 25, 28, 192), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 28, 3, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 25, 192), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>44</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 192), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>45</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 11, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 14, 384), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>46</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 14, 11, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 3, 384), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 3, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 11, 14, 384), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 14, 3, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 11, 384), dtype=float32)</td>
      <td>axis : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 384), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 3, 4, 4), dtype=float32)</td>
      <td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 384), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 768), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 1536), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td>Gelu</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 3072), dtype=float32)</td>
      <td>approximate : "none"</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(64, 3, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>56</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(64, 49, 96), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>57</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>58</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>59</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(16, 6, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>60</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(16, 49, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>61</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>62</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>63</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(4, 12, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(4, 49, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>65</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 1536), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>66</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>67</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 24, 49, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>68</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>69</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 3072), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>70</th>
      <td>Identity</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>71</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>72</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>73</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>74</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 3<br>stop : 56<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>75</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>76</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 3<br>stop : 56<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>77</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>78</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 53<br>stop : 56<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>79</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 53<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>80</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 53<br>stop : 56<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>81</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 53<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>82</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 56<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>83</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>dim : -3<br>start : 1<br>stop : 56<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>84</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 56<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>85</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 56, 96), dtype=float32)</td>
      <td>dim : -2<br>start : 1<br>stop : 56<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>86</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>87</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>88</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>89</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 3<br>stop : 28<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>90</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>91</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 3<br>stop : 28<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>92</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>93</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 25<br>stop : 28<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>94</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 25<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>95</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 25<br>stop : 28<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>96</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 25<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>97</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 28<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>98</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>dim : -3<br>start : 1<br>stop : 28<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>99</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 28<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>100</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 28, 192), dtype=float32)</td>
      <td>dim : -2<br>start : 1<br>stop : 28<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>101</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>102</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>103</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>104</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 3<br>stop : 14<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>105</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>106</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 3<br>stop : 14<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>107</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>108</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 11<br>stop : 14<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>109</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 11<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>110</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 11<br>stop : 14<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>111</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 11<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>112</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 0<br>stop : 14<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>dim : -3<br>start : 1<br>stop : 14<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>114</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 7, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 0<br>stop : 14<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>115</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 7, 14, 384), dtype=float32)</td>
      <td>dim : -2<br>start : 1<br>stop : 14<br>stride : 2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>116</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>117</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>118</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
      <td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>119</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>120</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>121</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>122</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>123</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>124</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>125</th>
      <td>Layernorm</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
      <td>dim : -1<br>epsilon : 1e-05</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>126</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1000), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>127</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 288), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>128</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(192, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 32, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>129</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(192, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 49, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>130</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 96), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>132</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(3136, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 96), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>133</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(784, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>134</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 576), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>135</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(96, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 32, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>136</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(96, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 49, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>137</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>138</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>139</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(784, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 192), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>140</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(196, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>141</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1152), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>142</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(48, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 32, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>143</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(48, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 49, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>144</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>145</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1536), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>146</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(196, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 384), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>147</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(49, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>148</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2304), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>149</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(24, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 32, 49), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>150</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(24, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 49, 32), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>151</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>152</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>153</th>
      <td>Matmul</td>
      <td>Operand(type=Activation, shape=(49, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(64, 3, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(16, 6, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(4, 12, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 24, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_150, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>158</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)</td>
      <td>shape : (1, 768, 1, 1)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>159</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>shape : (1, 8, 7, 8, 7, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>160</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
      <td>shape : (3136, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>161</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 8, 8, 7, 7, 96), dtype=float32)</td>
      <td>shape : (3136, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>162</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(3136, 288), dtype=float32)</td>
      <td>shape : (64, 49, 288)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>163</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(64, 49, 288), dtype=float32)</td>
      <td>shape : (64, 49, 3, 3, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>164</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 64, 3, 49, 32), dtype=float32)</td>
      <td>shape : (64, 3, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>165</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 64, 3, 49, 32), dtype=float32)</td>
      <td>shape : (192, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>166</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(64, 3, 49, 32), dtype=float32)</td>
      <td>shape : (192, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>167</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(192, 49, 49), dtype=float32)</td>
      <td>shape : (64, 3, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>168</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(2401, 3), dtype=float32)</td>
      <td>shape : (49, 49, 3)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>169</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(64, 3, 49, 49), dtype=float32)</td>
      <td>shape : (192, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>170</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(64, 3, 49, 49), dtype=float32)</td>
      <td>shape : (1, 64, 3, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>171</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(64, 3, 32, 49), dtype=float32)</td>
      <td>shape : (192, 32, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>172</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(192, 49, 32), dtype=float32)</td>
      <td>shape : (64, 3, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>173</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(64, 49, 3, 32), dtype=float32)</td>
      <td>shape : (3136, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>174</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(3136, 96), dtype=float32)</td>
      <td>shape : (64, 49, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>175</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(3136, 96), dtype=float32)</td>
      <td>shape : (1, 56, 56, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>176</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(64, 49, 96), dtype=float32)</td>
      <td>shape : (1, 8, 8, 7, 7, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>177</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 8, 7, 8, 7, 96), dtype=float32)</td>
      <td>shape : (1, 56, 56, 96)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>178</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(3136, 384), dtype=float32)</td>
      <td>shape : (1, 56, 56, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>179</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 56, 56, 384), dtype=float32)</td>
      <td>shape : (3136, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>180</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 64, 3, 49, 49), dtype=float32)</td>
      <td>shape : (64, 3, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>181</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 384), dtype=float32)</td>
      <td>shape : (784, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>182</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(784, 192), dtype=float32)</td>
      <td>shape : (1, 28, 28, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>183</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(784, 192), dtype=float32)</td>
      <td>shape : (16, 49, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>184</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>shape : (1, 4, 7, 4, 7, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>185</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
      <td>shape : (784, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>186</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 4, 4, 7, 7, 192), dtype=float32)</td>
      <td>shape : (784, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>187</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(784, 576), dtype=float32)</td>
      <td>shape : (16, 49, 576)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>188</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 49, 576), dtype=float32)</td>
      <td>shape : (16, 49, 3, 6, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>189</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 16, 6, 49, 32), dtype=float32)</td>
      <td>shape : (16, 6, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>190</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 16, 6, 49, 32), dtype=float32)</td>
      <td>shape : (96, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>191</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 6, 49, 32), dtype=float32)</td>
      <td>shape : (96, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>192</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(96, 49, 49), dtype=float32)</td>
      <td>shape : (16, 6, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>193</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(2401, 6), dtype=float32)</td>
      <td>shape : (49, 49, 6)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>194</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 6, 49, 49), dtype=float32)</td>
      <td>shape : (96, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>195</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 6, 49, 49), dtype=float32)</td>
      <td>shape : (1, 16, 6, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>196</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 6, 32, 49), dtype=float32)</td>
      <td>shape : (96, 32, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>197</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(96, 49, 32), dtype=float32)</td>
      <td>shape : (16, 6, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>198</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 49, 6, 32), dtype=float32)</td>
      <td>shape : (784, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>199</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(16, 49, 192), dtype=float32)</td>
      <td>shape : (1, 4, 4, 7, 7, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>200</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 4, 7, 4, 7, 192), dtype=float32)</td>
      <td>shape : (1, 28, 28, 192)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>201</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(784, 768), dtype=float32)</td>
      <td>shape : (1, 28, 28, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>202</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 28, 28, 768), dtype=float32)</td>
      <td>shape : (784, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>203</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 16, 6, 49, 49), dtype=float32)</td>
      <td>shape : (16, 6, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>204</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 768), dtype=float32)</td>
      <td>shape : (196, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>205</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(196, 384), dtype=float32)</td>
      <td>shape : (1, 14, 14, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>206</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(196, 384), dtype=float32)</td>
      <td>shape : (4, 49, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>207</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>shape : (1, 2, 7, 2, 7, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>208</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
      <td>shape : (196, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>209</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 2, 2, 7, 7, 384), dtype=float32)</td>
      <td>shape : (196, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>210</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(196, 1152), dtype=float32)</td>
      <td>shape : (4, 49, 1152)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>211</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(4, 49, 1152), dtype=float32)</td>
      <td>shape : (4, 49, 3, 12, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>212</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 4, 12, 49, 32), dtype=float32)</td>
      <td>shape : (4, 12, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>213</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 4, 12, 49, 32), dtype=float32)</td>
      <td>shape : (48, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>214</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(4, 12, 49, 32), dtype=float32)</td>
      <td>shape : (48, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>215</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(48, 49, 49), dtype=float32)</td>
      <td>shape : (4, 12, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>216</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(2401, 12), dtype=float32)</td>
      <td>shape : (49, 49, 12)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>217</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(4, 12, 49, 49), dtype=float32)</td>
      <td>shape : (48, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>218</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(4, 12, 49, 49), dtype=float32)</td>
      <td>shape : (1, 4, 12, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>219</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(4, 12, 32, 49), dtype=float32)</td>
      <td>shape : (48, 32, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>220</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(48, 49, 32), dtype=float32)</td>
      <td>shape : (4, 12, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>221</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(4, 49, 12, 32), dtype=float32)</td>
      <td>shape : (196, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>222</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(4, 49, 384), dtype=float32)</td>
      <td>shape : (1, 2, 2, 7, 7, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>223</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 2, 7, 2, 7, 384), dtype=float32)</td>
      <td>shape : (1, 14, 14, 384)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>224</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(196, 1536), dtype=float32)</td>
      <td>shape : (1, 14, 14, 1536)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>225</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 14, 14, 1536), dtype=float32)</td>
      <td>shape : (196, 1536)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>226</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 4, 12, 49, 49), dtype=float32)</td>
      <td>shape : (4, 12, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>227</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 1536), dtype=float32)</td>
      <td>shape : (49, 1536)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>228</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(49, 768), dtype=float32)</td>
      <td>shape : (1, 7, 7, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>229</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(49, 768), dtype=float32)</td>
      <td>shape : (1, 49, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>230</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)</td>
      <td>shape : (1, 1, 7, 1, 7, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>231</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)</td>
      <td>shape : (49, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>232</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 1, 7, 7, 768), dtype=float32)</td>
      <td>shape : (49, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>233</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(49, 2304), dtype=float32)</td>
      <td>shape : (1, 49, 2304)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>234</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 49, 2304), dtype=float32)</td>
      <td>shape : (1, 49, 3, 24, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>235</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 24, 49, 32), dtype=float32)</td>
      <td>shape : (1, 24, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>236</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 24, 49, 32), dtype=float32)</td>
      <td>shape : (24, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>237</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 24, 49, 32), dtype=float32)</td>
      <td>shape : (24, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>238</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(24, 49, 49), dtype=float32)</td>
      <td>shape : (1, 24, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>239</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(2401, 24), dtype=float32)</td>
      <td>shape : (49, 49, 24)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>240</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 24, 49, 49), dtype=float32)</td>
      <td>shape : (24, 49, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>241</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 24, 32, 49), dtype=float32)</td>
      <td>shape : (24, 32, 49)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>242</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(24, 49, 32), dtype=float32)</td>
      <td>shape : (1, 24, 49, 32)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>243</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 49, 24, 32), dtype=float32)</td>
      <td>shape : (49, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>244</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)</td>
      <td>shape : (1, 1, 1, 7, 7, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>245</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 7, 1, 7, 768), dtype=float32)</td>
      <td>shape : (1, 7, 7, 768)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>246</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(49, 3072), dtype=float32)</td>
      <td>shape : (1, 7, 7, 3072)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>247</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 3072), dtype=float32)</td>
      <td>shape : (49, 3072)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>248</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(64, 3, 49, 49), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>249</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(16, 6, 49, 49), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>250</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(4, 12, 49, 49), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>251</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 24, 49, 49), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>252</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)</td>
      <td>dim : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>253</th>
      <td>Squeeze</td>
      <td>Operand(type=Activation, shape=(1, 768, 1), dtype=float32)</td>
      <td>dim : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>254</th>
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
      <th>255</th>
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
      <th>256</th>
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
      <th>257</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(384, 384), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>258</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1536, 384), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>259</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(384, 1536), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>260</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1000, 768), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>261</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(192, 192), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>262</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(768, 192), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>263</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(192, 768), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>264</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>265</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 56, 96, 56), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>266</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 8, 7, 8, 7, 96), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>267</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(288, 96), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>268</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(64, 49, 3, 3, 32), dtype=float32)</td>
      <td>dim0 : -5<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>269</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 49, 64, 3, 32), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>270</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 64, 49, 3, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>271</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(192, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>272</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(49, 49, 3), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>273</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 49, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>274</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(64, 3, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>275</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(64, 3, 49, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>276</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(192, 32, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>277</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(96, 96), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>278</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 8, 8, 7, 7, 96), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>279</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(384, 96), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>280</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(96, 384), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>281</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(192, 384), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>282</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 4, 7, 4, 7, 192), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>283</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(576, 192), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>284</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(16, 49, 3, 6, 32), dtype=float32)</td>
      <td>dim0 : -5<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>285</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 49, 16, 6, 32), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>286</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 16, 49, 6, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>287</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(96, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>288</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(49, 49, 6), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>289</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(6, 49, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>290</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(16, 6, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>291</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(16, 6, 49, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>292</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(96, 32, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>293</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 4, 4, 7, 7, 192), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>294</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(384, 768), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>295</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 2, 7, 2, 7, 384), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>296</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(1152, 384), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>297</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(4, 49, 3, 12, 32), dtype=float32)</td>
      <td>dim0 : -5<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>298</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 49, 4, 12, 32), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>299</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 4, 49, 12, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>300</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(48, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>301</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(49, 49, 12), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>302</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(12, 49, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>303</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(4, 12, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>304</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(4, 12, 49, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>305</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(48, 32, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>306</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 2, 2, 7, 7, 384), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>307</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(768, 1536), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>308</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 1, 7, 1, 7, 768), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>309</th>
      <td>Transpose</td>
      <td>Operand(type=Parameter, shape=(2304, 768), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>310</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 49, 3, 24, 32), dtype=float32)</td>
      <td>dim0 : -5<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>311</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 49, 1, 24, 32), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>312</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(3, 1, 49, 24, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>313</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(24, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>314</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(49, 49, 24), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>315</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(24, 49, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>316</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 24, 49, 32), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>317</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 24, 49, 32), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>318</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(24, 32, 49), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>319</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 1, 1, 7, 7, 768), dtype=float32)</td>
      <td>dim0 : -4<br>dim1 : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>320</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>321</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>322</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>323</th>
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
      <th>324</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(3, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>325</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(6, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>326</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(12, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>327</th>
      <td>Unsqueeze</td>
      <td>Operand(type=Activation, shape=(24, 49, 49), dtype=float32)</td>
      <td>dim : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
