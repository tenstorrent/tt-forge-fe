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
			<td rowspan="3">1</td>
			<td rowspan="3">[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: cumsum</td>
			<td rowspan="3">11</td>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)</td>
			<td>axis : 1<br>exclusive : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)</td>
			<td>axis : 1<br>exclusive : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int32)</td>
			<td>axis : 1<br>exclusive : 0</td>
		</tr>
	</tbody>
</table>
