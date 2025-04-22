<h1>Comprehensive Report on Subtract Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of subtract operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Subtract Operation Details</th>
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
			<td rowspan="6">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="6">19</td>
			<td>8</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_70, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_2115, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_70, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_60, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_2115, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 384), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_1940, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="4">2</td>
			<td rowspan="4">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="4">15</td>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1,), dtype=int32)<br><div align='center'>X</div>Operand(type=Constant, name=const_520, dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_2445, dtype=int64)</td>
			<td></td>
		</tr>
	</tbody>
</table>
