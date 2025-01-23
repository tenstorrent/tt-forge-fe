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
			<td rowspan="18">1</td>
			<td rowspan="18">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td rowspan="18">19</td>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50257, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_base_v1_mlm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(30000, 128), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_base_v1_mlm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 128), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_base_v1_mlm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 128), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50265, 1024), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(1026, 1024), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(30522, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50272, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2050, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(151936, 1024), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(250002, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(514, 768), dtype=bfloat16)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(256008, 1024), dtype=bfloat16)</td>
			<td></td>
		</tr>
	</tbody>
</table>
