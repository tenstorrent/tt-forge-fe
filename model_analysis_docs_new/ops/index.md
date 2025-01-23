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
			<td>Operand(type=Constant, name=albert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td>2</td>
			<td><ul><li>pt_albert_base_v2_mlm</li><li>pt_albert_base_v1_mlm</li></ul></td>
		</tr>
		<tr>
			<td>2</td>
			<td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td>4</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li><li>pt_albert_base_v2_mlm</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder</li><li>pt_albert_base_v1_mlm</li></ul></td>
		</tr>
		<tr>
			<td>3</td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 768<br>stride : 1</td>
			<td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
		</tr>
		<tr>
			<td>4</td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 768<br>stop : 1536<br>stride : 1</td>
			<td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
		</tr>
		<tr>
			<td>5</td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 1536<br>stop : 2304<br>stride : 1</td>
			<td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
		</tr>
		<tr>
			<td>6</td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
			<td>[FORGE][Runtime Datatype Unsupported] RuntimeError Unhandled dtype Bool</td>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
		</tr>
		<tr>
			<td>7</td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
			<td>[FORGE][Runtime Datatype Unsupported] RuntimeError Unhandled dtype Bool</td>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li></ul></td>
		</tr>
		<tr>
			<td>8</td>
			<td>Operand(type=Constant, name=model.transformer.h.1.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
			<td>[FORGE][Runtime Datatype Unsupported] RuntimeError Unhandled dtype Bool</td>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li></ul></td>
		</tr>
		<tr>
			<td>9</td>
			<td>Operand(type=Constant, name=roberta.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
	<tbody>
</table>
