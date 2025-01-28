<h1>Comprehensive Report on Greater Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of greater operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Greater Operation Details</th>
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
			<td rowspan="2">1</td>
			<td rowspan="2">[TT_METAL][ttnn elementwise binary] RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
			<td rowspan="2">3</td>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_80, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60, dtype=float32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
