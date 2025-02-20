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
			<td rowspan="6">1</td>
			<td rowspan="6">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="6">11</td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_3153, dtype=int32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 12), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1026, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39), dtype=int64)<br><div align='center'>X</div>Operand(type=Parameter, shape=(102400, 4096), dtype=bfloat16)</td>
			<td></td>
		</tr>
	</tbody>
</table>
