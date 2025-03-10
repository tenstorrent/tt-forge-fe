<h1>Comprehensive Report on Broadcast Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of broadcast operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Broadcast Operation Details</th>
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
			<td rowspan="5">1</td>
			<td rowspan="5">[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td rowspan="5">13</td>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 128), dtype=uint1)</td>
			<td>dim : -3<br>shape : 12</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 128), dtype=uint1)</td>
			<td>dim : -2<br>shape : 128</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596, 1), dtype=uint1)</td>
			<td>dim : -1<br>shape : 4096</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 384), dtype=uint1)</td>
			<td>dim : -3<br>shape : 12</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 384), dtype=uint1)</td>
			<td>dim : -2<br>shape : 384</td>
		</tr>
		<tr>
			<td rowspan="2">2</td>
			<td rowspan="2">[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 0 - data type mismatch: expected UInt8, got Float32</td>
			<td rowspan="2">6</td>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 128), dtype=uint1)</td>
			<td>dim : -4<br>shape : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 384), dtype=uint1)</td>
			<td>dim : -4<br>shape : 1</td>
		</tr>
	</tbody>
</table>
