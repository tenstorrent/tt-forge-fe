<h1>Comprehensive Report on Stack Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of stack operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Stack Operation Details</th>
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
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 1, 2048), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
	</tbody>
</table>
