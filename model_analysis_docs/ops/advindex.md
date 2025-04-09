<h1>Comprehensive Report on AdvIndex Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of advindex operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Advindex Operation Details</th>
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
			<td rowspan="8">1</td>
			<td rowspan="8">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="8">10</td>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2359296,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2441216,), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2401,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2401,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2401,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2401,), dtype=int64)</td>
			<td></td>
		</tr>
	</tbody>
</table>
