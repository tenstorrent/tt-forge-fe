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
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)</td>
			<td>axis : 1<br>exclusive : 0</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: cumsum</td>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_opt_facebook_opt_350m_clm</li></ul></td>
		</tr>
		<tr>
			<td>2</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int32)</td>
			<td>axis : 1<br>exclusive : 0</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: cumsum</td>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
	<tbody>
</table>
