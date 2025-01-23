<h1>Comprehensive Report on RepeatInterleave Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of repeatinterleave operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Repeatinterleave Operation Details</th>
		</tr>
		<tr style="text-align: center;">
			<th>ID</th>
			<th>Failure Description</th>
			<th>Total Number of Models Affected</th>
			<th>Number of Models Affected</th>
			<th>Affected Models</th>
			<th>Operands</th>
			<th>Arguments</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td rowspan="6">1</td>
			<td rowspan="6">[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td rowspan="6">13</td>
			<td>3</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 256<br>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_albert_base_v1_mlm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 2</td>
		</tr>
	</tbody>
</table>
