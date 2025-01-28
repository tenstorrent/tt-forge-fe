<h1>Comprehensive Report on Index Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of index operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Index Operation Details</th>
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
			<td rowspan="3">1</td>
			<td rowspan="3">[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td rowspan="3">4</td>
			<td>2</td>
			<td><ul><li>pt_albert_base_v1_mlm</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_base_v1_mlm</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
			<td>Operand(type=Constant, name=roberta.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td rowspan="3">2</td>
			<td rowspan="3">[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td rowspan="3">3</td>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 768<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 1536<br>stop : 2304<br>stride : 1</td>
		</tr>
		<tr>
			<td rowspan="3">3</td>
			<td rowspan="3">[FORGE][Runtime Datatype Unsupported] RuntimeError Unhandled dtype Bool</td>
			<td rowspan="3">3</td>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.1.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
	</tbody>
</table>
