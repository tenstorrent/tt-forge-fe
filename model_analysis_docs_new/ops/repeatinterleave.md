<h1>Comprehensive Report on Operation Failures and Affected Models</h1>
<p>This table provides detailed insights into operation specific statistics, highlighting the number of failed models for each operation and the associated models that encountered issues. Click on an Operation name to view its detailed analysis</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="3">Operation Details</th>
			<th colspan="3">Failure Insight and Impacted Models</th>
		</tr>
		<tr style="text-align: center;">
			<th>ID</th>
			<th>Operands</th>
			<th>Arguments</th>
			<th>Failure</th>
			<th>Number of Models Affected</th>
			<th>Affected Models</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>1</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>3</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_albert_base_v2_mlm</li><li>pt_albert_base_v1_mlm</li></ul></td>
		</tr>
		<tr>
			<td>2</td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>5</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_opt_facebook_opt_350m_clm</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
		</tr>
		<tr>
			<td>3</td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>5</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_opt_facebook_opt_350m_clm</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
		</tr>
		<tr>
			<td>4</td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 256<br>dim : 2</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>5</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_opt_facebook_opt_350m_clm</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
		</tr>
		<tr>
			<td>5</td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm</li></ul></td>
		</tr>
		<tr>
			<td>6</td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 2</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm</li></ul></td>
		</tr>
		<tr>
			<td>7</td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 35, 64), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</li></ul></td>
		</tr>
		<tr>
			<td>8</td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 35, 64), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</li></ul></td>
		</tr>
		<tr>
			<td>9</td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 29, 64), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm</li></ul></td>
		</tr>
		<tr>
			<td>10</td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 29, 64), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm</li></ul></td>
		</tr>
	<tbody>
</table>
