<h1>Comprehensive Report on CumSum Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of cumsum operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Cumsum Operation Details</th>
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
			<td rowspan="2">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td rowspan="2">9</td>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="1">2</td>
			<td>2</td>
			<td><ul><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td rowspan="1">3</td>
			<td rowspan="1">[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2441216,), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
	</tbody>
</table>
