<h1>Comprehensive Report on Remainder Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of remainder operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Remainder Operation Details</th>
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
			<td rowspan="1">1</td>
			<td rowspan="1">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="1">3</td>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1,), dtype=int32)<br><div align='center'>X</div>Operand(type=Constant, name=const_600, dtype=int32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
