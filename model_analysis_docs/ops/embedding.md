<h1>Comprehensive Report on Embedding Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of embedding operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Embedding Operation Details</th>
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
			<td rowspan="4">1</td>
			<td rowspan="4">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="4">7</td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_3153, dtype=int32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 12), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_40, dtype=int32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 16), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_40, dtype=int32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 8), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_40, dtype=int32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 6), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="1">2</td>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_40, dtype=int32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 12), dtype=float32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
