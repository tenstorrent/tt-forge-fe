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
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
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
      <td>Operand(type=Constant, name=const_0209, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>10</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>11</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>12</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>13</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>14</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>15</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>16</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>17</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>18</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>19</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 96, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 28, 40), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>20</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 192, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 20), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>21</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
      <td>axis : -3</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>22</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 4, 4480), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 1120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 280), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>23</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>24</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 80, 4480), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 1120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 280), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>25</th>
      <td>Concatenate</td>
      <td>Operand(type=Activation, shape=(1, 5880, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_208209, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 80), dtype=float32)</td>
      <td>axis : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>26</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 3, 448, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2209, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 48, 224, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4209, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>28</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>29</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>30</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>31</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_12209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>32</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_14209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>33</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_15209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_16209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>34</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_17209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_18209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>35</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_19209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20209, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>36</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_21209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>37</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_24209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>38</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>39</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_27209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_28209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>40</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>41</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_32209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>42</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_33209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_34209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>43</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_36209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>44</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>45</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_39209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40209, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>46</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_42209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>47</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_43209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_44209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>48</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_45209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_46209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>49</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_48209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>50</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_49209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>51</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_51209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_52209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>52</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_54209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>53</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>54</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_57209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>55</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>56</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_61209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_62209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>57</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_63209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>58</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_65209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_66209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>59</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_68209, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>60</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_69209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>61</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_71209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_72209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>62</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_74209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>63</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_75209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>64</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_78209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>65</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>66</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1024, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_81209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>67</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_84209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>68</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 1536, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_86209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>69</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_87209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>70</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>71</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_93209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_94209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_95209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_96209, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>73</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 576, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_97209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_98209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>74</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_99209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>75</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_102209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>76</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_104209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>77</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_105209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_106209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_107209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_108209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_109209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>80</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_111209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_112209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_113209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_114209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>82</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_115209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_117209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_118209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_121209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_122209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 96, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_123209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_124209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>86</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_126209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 288, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_127209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>88</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_129209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_130209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_132209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>90</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_133209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_134209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>91</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_135209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_136209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>92</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_137209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_138209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_139209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_140209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_141209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_142209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_143209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_144209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>96</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_145209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_146209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>97</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_147209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_148209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>98</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_149209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_150209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>99</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_151209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_152209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 17, 4, 4480), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_153209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_154209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_155209, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>102</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_156209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_157209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>103</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_158209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_159209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>104</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_160209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_161209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>105</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_162209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_163209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>106</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_164209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_165209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>107</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_166209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_167209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>108</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_168209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_169209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>109</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_170209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_171209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>110</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_172209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_173209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_174209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_175209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_176209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_177209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>113</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_178209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_179209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 17, 4, 1120), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_153209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_180209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_181209, dtype=float32)</td>
      <td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>116</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_182209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_183209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>117</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_184209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_185209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>118</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_186209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_187209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>119</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_188209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_189209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>120</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_190209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_191209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>121</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_192209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_193209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>122</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_194209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_195209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>123</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_196209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_197209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_198209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_199209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>125</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_200209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_201209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>126</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_202209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_203209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>127</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_204209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_205209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 17, 4, 280), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_153209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>129</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_209209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_210209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>130</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_211209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_212209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_213209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_214209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>132</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_215209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_216209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>133</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_217209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_218209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>134</th>
      <td>Conv2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_219209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_220209, dtype=float32)</td>
      <td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>135</th>
      <td>Conv2dTranspose</td>
      <td>Operand(type=Activation, shape=(1, 192, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_90209, dtype=float32)</td>
      <td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: conv2d_transpose</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Conv2dTranspose</td>
      <td>Operand(type=Activation, shape=(1, 96, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_120209, dtype=float32)</td>
      <td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: conv2d_transpose</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 5880, 4), dtype=float32)</td>
      <td>dim : -1<br>start : 0<br>stop : 2<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>138</th>
      <td>Index</td>
      <td>Operand(type=Activation, shape=(1, 5880, 4), dtype=float32)</td>
      <td>dim : -1<br>start : 2<br>stop : 4<br>stride : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>139</th>
      <td>MaxPool2d</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)</td>
      <td>kernel_size : 5<br>stride : 1<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>ceil_mode : False<br>max_pool_add_sub_surround : False<br>max_pool_add_sub_surround_value : 1.0<br>channel_last : 0</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>140</th>
      <td>Multiply</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_206209, dtype=float32)</td>
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
      <td>Operand(type=Activation, shape=(1, 5880, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_207209, dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>145</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 48, 224, 320), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>146</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 96, 112, 160), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>147</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 64, 112, 160), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>148</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 192, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>149</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 128, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>150</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 384, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>151</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 256, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>152</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 768, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>153</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 512, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>154</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>155</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 192, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>156</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>157</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>158</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 96, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>159</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>160</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 64, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>161</th>
      <td>Relu</td>
      <td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>162</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 68, 56, 80), dtype=float32)</td>
      <td>shape : (1, 4, 17, 4480)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>163</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 4, 4480), dtype=float32)</td>
      <td>shape : (1, 4, 4480)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>164</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 68, 28, 40), dtype=float32)</td>
      <td>shape : (1, 4, 17, 1120)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>165</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 4, 1120), dtype=float32)</td>
      <td>shape : (1, 4, 1120)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>166</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 68, 14, 20), dtype=float32)</td>
      <td>shape : (1, 4, 17, 280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>167</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 1, 4, 280), dtype=float32)</td>
      <td>shape : (1, 4, 280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>168</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 80, 56, 80), dtype=float32)</td>
      <td>shape : (1, 80, 4480)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>169</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 80, 28, 40), dtype=float32)</td>
      <td>shape : (1, 80, 1120)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>170</th>
      <td>Reshape</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 20), dtype=float32)</td>
      <td>shape : (1, 80, 280)</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>171</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 96, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>172</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 192, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>173</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 384, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>174</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 80, 56, 80), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>175</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 80, 28, 40), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>176</th>
      <td>Sigmoid</td>
      <td>Operand(type=Activation, shape=(1, 80, 14, 20), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>177</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 17, 4, 4480), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn softmax] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B Inputs must be of bfloat16 or bfloat8_b type</td>
    </tr>
    <tr>
      <th>178</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 17, 4, 1120), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn softmax] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B Inputs must be of bfloat16 or bfloat8_b type</td>
    </tr>
    <tr>
      <th>179</th>
      <td>Softmax</td>
      <td>Operand(type=Activation, shape=(1, 17, 4, 280), dtype=float32)</td>
      <td>dim : 1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x274C;</td>
      <td></td>
      <td>[TT_METAL][ttnn softmax] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B Inputs must be of bfloat16 or bfloat8_b type</td>
    </tr>
    <tr>
      <th>180</th>
      <td>Subtract</td>
      <td>Operand(type=Constant, name=const_0209, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>181</th>
      <td>Subtract</td>
      <td>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5880, 2), dtype=float32)</td>
      <td></td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>182</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 4, 17, 4480), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>183</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 4, 17, 1120), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>184</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 4, 17, 280), dtype=float32)</td>
      <td>dim0 : -3<br>dim1 : -2</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>185</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 4, 5880), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>186</th>
      <td>Transpose</td>
      <td>Operand(type=Activation, shape=(1, 80, 5880), dtype=float32)</td>
      <td>dim0 : -2<br>dim1 : -1</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td>&#x2705;</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
