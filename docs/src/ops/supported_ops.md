# Supported Operators

## Forge Operators

| Name | Full Name / Path | Inputs | Constructor Parameters | Forward Parameters |
|------|--------------------|--------|------------------------|--------------------|
| exp | forge.op.Exp | 1-1 | N/A | N/A |
| reciprocal | forge.op.Reciprocal | 1-1 | N/A | N/A |
| buffer | forge.op.Buffer | 1-1 | N/A | N/A |
| sqrt | forge.op.Sqrt | 1-1 | N/A | N/A |
| relu | forge.op.Relu | 1-1 | N/A | N/A |
| leaky_relu | forge.op.LeakyRelu | 1-1 | N/A | alpha (<class 'float'>): [0, 100] |
| nop | forge.op.Identity | 1-1 | N/A | N/A |
| gelu | forge.op.Gelu | 1-1 | N/A | N/A |
| log | forge.op.Log | 1-1 | N/A | N/A |
| sigmoid | forge.op.Sigmoid | 1-1 | N/A | N/A |
| clip | forge.op.Clip | 1-1 | N/A | min (<class 'float'>): [0, 100], max (<class 'float'>): [0, 100] |
| sine | forge.op.Sine | 1-1 | N/A | N/A |
| cosine | forge.op.Cosine | 1-1 | N/A | N/A |
| abs | forge.op.Abs | 1-1 | N/A | N/A |
| tanh | forge.op.Tanh | 1-1 | N/A | N/A |
| cumsum | forge.op.CumSum | 1-1 | N/A | N/A |
| argmax | forge.op.Argmax | 1-1 | N/A | N/A |
| logical_not | forge.op.LogicalNot | 1-1 | N/A | N/A |
| dropout | forge.op.Dropout | 1-1 | N/A | N/A |
| pow | forge.op.Pow | 1-1 | N/A | exponent (<class 'float'>): [0, 100] |
| tilizer | forge.op.Tilize | 1-1 | N/A | N/A |
| add | forge.op.Add | 2-2 | N/A | N/A |
| divide | forge.op.Divide | 2-2 | N/A | N/A |
| subtract | forge.op.Subtract | 2-2 | N/A | N/A |
| multiply | forge.op.Multiply | 2-2 | N/A | N/A |
| maximum | forge.op.Max | 2-2 | N/A | N/A |
| minimum | forge.op.Min | 2-2 | N/A | N/A |
| heaviside | forge.op.Heaviside | 2-2 | N/A | N/A |
| power | forge.op.Power | 2-2 | N/A | N/A |
| greater | forge.op.Greater | 2-2 | N/A | N/A |
| greater_equal | forge.op.GreaterEqual | 2-2 | N/A | N/A |
| less | forge.op.Less | 2-2 | N/A | N/A |
| less_equal | forge.op.LessEqual | 2-2 | N/A | N/A |
| equal | forge.op.Equal | 2-2 | N/A | N/A |
| not_equal | forge.op.NotEqual | 2-2 | N/A | N/A |
| logical_and | forge.op.LogicalAnd | 2-2 | N/A | N/A |
| where | forge.op.Where | 3-3 | N/A | N/A |
| interleave | forge.op.Interleave | 1-1 | N/A | axis (<class 'int'>): [-3, -3], stride (<class 'int'>): [1, 1] |
| concatenate | forge.op.Concatenate | 1-1 | N/A | axis (<class 'int'>): [-10, 10] |
| stack | forge.op.Stack | 2-2 | N/A | axis (<class 'int'>): [1, 10] |
| matmul | forge.op.Matmul | 2-2 | N/A | N/A |

## PyTorch Operators

| Name | Full Name / Path | Inputs | Constructor Parameters | Forward Parameters |
|------|--------------------|--------|------------------------|--------------------|
| embedding | torch.nn.Embedding | 1-1 | N/A | N/A |
| linear | torch.nn.Linear | 1-1 | in_features (<class 'int'>): [10, 50], out_features (<class 'int'>): [10, 50] | N/A |
| conv2d | torch.nn.Conv2d | 1-1 | in_channels (<class 'int'>): [10, 50], out_channels (<class 'int'>): [10, 50], kernel_size (<class 'int'>): [3, 3], stride (<class 'int'>): [1, 1], padding (<class 'int'>): [1, 1] | N/A |
| relu | torch.relu | 1-1 | N/A | N/A |
| sqrt | torch.sqrt | 1-1 | N/A | N/A |
| reciprocal | torch.reciprocal | 1-1 | N/A | N/A |
| sigmoid | torch.sigmoid | 1-1 | N/A | N/A |
| abs | torch.abs | 1-1 | N/A | N/A |
| cos | torch.cos | 1-1 | N/A | N/A |
| exp | torch.exp | 1-1 | N/A | N/A |
| neg | torch.neg | 1-1 | N/A | N/A |
| rsqrt | torch.rsqrt | 1-1 | N/A | N/A |
| sin | torch.sin | 1-1 | N/A | N/A |
| square | torch.square | 1-1 | N/A | N/A |
| pow | torch.pow | 1-1 | N/A | exponent (<class 'int'>): [-10, 10] |
| clamp | torch.clamp | 1-1 | N/A | min (<class 'int'>): [-100, 100], max (<class 'int'>): [-100, 100] |
| log | torch.log | 1-1 | N/A | N/A |
| log1p | torch.log1p | 1-1 | N/A | N/A |
| gelu | torch.nn.functional.gelu | 1-1 | N/A | N/A |
| leaky_relu | torch.nn.functional.leaky_relu | 1-1 | N/A | N/A |
| cumsum | torch.cumsum | 1-1 | N/A | dim (<class 'int'>): [-3, 3] |
| softmax | torch.softmax | 1-1 | N/A | N/A |
| acos | torch.acos | 1-1 | N/A | N/A |
| arccos | torch.acos | 1-1 | N/A | N/A |
| acosh | torch.acosh | 1-1 | N/A | N/A |
| arccosh | torch.acosh | 1-1 | N/A | N/A |
| angle | torch.angle | 1-1 | N/A | N/A |
| asin | torch.asin | 1-1 | N/A | N/A |
| arcsin | torch.asin | 1-1 | N/A | N/A |
| asinh | torch.asinh | 1-1 | N/A | N/A |
| arcsinh | torch.asinh | 1-1 | N/A | N/A |
| atan | torch.atan | 1-1 | N/A | N/A |
| arctan | torch.atan | 1-1 | N/A | N/A |
| atanh | torch.atanh | 1-1 | N/A | N/A |
| arctanh | torch.atanh | 1-1 | N/A | N/A |
| bitwise_not | torch.bitwise_not | 1-1 | N/A | N/A |
| ceil | torch.ceil | 1-1 | N/A | N/A |
| conj_physical | torch.conj_physical | 1-1 | N/A | N/A |
| cosh | torch.cosh | 1-1 | N/A | N/A |
| deg2rad | torch.deg2rad | 1-1 | N/A | N/A |
| digamma | torch.digamma | 1-1 | N/A | N/A |
| erf | torch.erf | 1-1 | N/A | N/A |
| erfc | torch.erfc | 1-1 | N/A | N/A |
| erfinv | torch.erfinv | 1-1 | N/A | N/A |
| exp2 | torch.exp2 | 1-1 | N/A | N/A |
| expm1 | torch.expm1 | 1-1 | N/A | N/A |
| fix | torch.fix | 1-1 | N/A | N/A |
| floor | torch.floor | 1-1 | N/A | N/A |
| frac | torch.frac | 1-1 | N/A | N/A |
| lgamma | torch.lgamma | 1-1 | N/A | N/A |
| log10 | torch.log10 | 1-1 | N/A | N/A |
| log2 | torch.log2 | 1-1 | N/A | N/A |
| logit | torch.logit | 1-1 | N/A | N/A |
| i0 | torch.i0 | 1-1 | N/A | N/A |
| isnan | torch.isnan | 1-1 | N/A | N/A |
| nan_to_num | torch.nan_to_num | 1-1 | N/A | N/A |
| positive | torch.positive | 1-1 | N/A | N/A |
| rad2deg | torch.rad2deg | 1-1 | N/A | N/A |
| round | torch.round | 1-1 | N/A | N/A |
| sign | torch.sign | 1-1 | N/A | N/A |
| sgn | torch.sgn | 1-1 | N/A | N/A |
| signbit | torch.signbit | 1-1 | N/A | N/A |
| sinc | torch.sinc | 1-1 | N/A | N/A |
| sinh | torch.sinh | 1-1 | N/A | N/A |
| tan | torch.tan | 1-1 | N/A | N/A |
| tanh | torch.tanh | 1-1 | N/A | N/A |
| trunc | torch.trunc | 1-1 | N/A | N/A |
| add | torch.add | 2-2 | N/A | N/A |
| sub | torch.sub | 2-2 | N/A | N/A |
| mul | torch.mul | 2-2 | N/A | N/A |
| div | torch.div | 2-2 | N/A | N/A |
| ge | torch.ge | 2-2 | N/A | N/A |
| ne | torch.ne | 2-2 | N/A | N/A |
| gt | torch.gt | 2-2 | N/A | N/A |
| lt | torch.lt | 2-2 | N/A | N/A |
| maximum | torch.maximum | 2-2 | N/A | N/A |
| minimum | torch.minimum | 2-2 | N/A | N/A |
| atan2 | torch.atan2 | 2-2 | N/A | N/A |
| arctan2 | torch.arctan2 | 2-2 | N/A | N/A |
| bitwise_and | torch.bitwise_and | 2-2 | N/A | N/A |
| bitwise_or | torch.bitwise_or | 2-2 | N/A | N/A |
| bitwise_xor | torch.bitwise_xor | 2-2 | N/A | N/A |
| bitwise_left_shift | torch.bitwise_left_shift | 2-2 | N/A | N/A |
| bitwise_right_shift | torch.bitwise_right_shift | 2-2 | N/A | N/A |
| floor_divide | torch.floor_divide | 2-2 | N/A | N/A |
| fmod | torch.fmod | 2-2 | N/A | N/A |
| logaddexp | torch.logaddexp | 2-2 | N/A | N/A |
| logaddexp2 | torch.logaddexp2 | 2-2 | N/A | N/A |
| nextafter | torch.nextafter | 2-2 | N/A | N/A |
| remainder | torch.remainder | 2-2 | N/A | N/A |
| fmax | torch.fmax | 2-2 | N/A | N/A |
| fmin | torch.fmin | 2-2 | N/A | N/A |
| eq | torch.eq | 2-2 | N/A | N/A |
| le | torch.le | 2-2 | N/A | N/A |
| matmul | torch.matmul | 2-2 | N/A | N/A |
| concatenate | torch.concatenate | 2-2 | N/A | N/A |
| max | torch.max | 1-1 | N/A | N/A |
| sum | torch.sum | 1-1 | N/A | N/A |
| mean | torch.mean | 1-1 | N/A | N/A |
| repeat_interleave | torch.repeat_interleave | 1-1 | N/A | N/A |
| reshape | torch.reshape | 1-1 | N/A | N/A |
| squeeze | torch.squeeze | 1-1 | N/A | N/A |
| unsqueeze | torch.unsqueeze | 1-1 | N/A | N/A |
| transpose | torch.transpose | 1-1 | N/A | N/A |
| layer_norm | torch.nn.LayerNorm | 1-1 | N/A | N/A |

