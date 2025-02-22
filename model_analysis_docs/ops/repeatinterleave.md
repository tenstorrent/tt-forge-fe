<h1>Comprehensive Report on RepeatInterleave Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of repeatinterleave operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Repeatinterleave Operation Details</th>
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
			<td rowspan="11">1</td>
			<td rowspan="11">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td rowspan="11">62</td>
			<td>23</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 32), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 32), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 32), dtype=int64)</td>
			<td>repeats : 32<br>dim : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1, 13), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1, 13), dtype=int64)</td>
			<td>repeats : 1<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1, 7), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1, 7), dtype=int64)</td>
			<td>repeats : 7<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[TT_METAL][tt-metal kernel] RuntimeError tt-metal/tt_metal/impl/kernels/kernel.cpp unique+common runtime args targeting kernel are too large</td>
			<td rowspan="1">6</td>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 256<br>dim : 2</td>
		</tr>
	</tbody>
</table>
