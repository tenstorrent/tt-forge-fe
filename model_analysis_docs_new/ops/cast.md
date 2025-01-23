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
			<td>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=uint1)</td>
			<td>dtype : torch.bool</td>
			<td>[MLIR][MLIR runtime ttnn ] tt::exception tt-mlir/runtime/lib/ttnn/runtime.cpp Unsupported data type</td>
			<td>4</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_gpt2_gpt2_text_gen</li></ul></td>
		</tr>
	<tbody>
</table>
