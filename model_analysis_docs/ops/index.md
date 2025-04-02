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
			<td rowspan="475">1</td>
			<td rowspan="475">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="475">958</td>
			<td>31</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=bert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>21</td>
			<td><ul><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=bert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2,), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2,), dtype=float32)</td>
			<td>dim : -1<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 2), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 4<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 4, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 4, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 18<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 18<br>stop : 54<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 54<br>stop : 126<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 72<br>stop : 108<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 108<br>stop : 126<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 36<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 36<br>stop : 54<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 54<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 18<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 18<br>stop : 36<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 36<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(16384, 3072), dtype=float32)</td>
			<td>dim : -2<br>start : 8192<br>stop : 16384<br>stride : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(16384, 3072), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 8192<br>stride : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=bert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1), dtype=int64)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1), dtype=int64)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1), dtype=int64)</td>
			<td>dim : -2<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1), dtype=int64)</td>
			<td>dim : -2<br>start : 3<br>stop : 4<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 768<br>stop : 1024<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1536<br>stop : 1792<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 2304<br>stop : 2560<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 512<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1280<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 2048<br>stop : 2304<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 2816<br>stop : 3072<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 256<br>stop : 512<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1024<br>stop : 1280<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1792<br>stop : 2048<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 2560<br>stop : 2816<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 1<br>stop : 32<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2304, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2304, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 768<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2304, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 1536<br>stop : 2304<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 768<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 1536<br>stop : 2304<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 32, 2048), dtype=uint1)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.1.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.1.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 2), dtype=float32)</td>
			<td>dim : -2<br>start : 31<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 2048), dtype=uint1)</td>
			<td>dim : -1<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024, 72), dtype=float32)</td>
			<td>dim : -1<br>start : -1<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 35, 35), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 35, 35), dtype=float32)</td>
			<td>dim : -3<br>start : 96<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 35, 35), dtype=float32)</td>
			<td>dim : -3<br>start : 160<br>stop : 224<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 17, 17), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 17, 17), dtype=float32)</td>
			<td>dim : -3<br>start : 384<br>stop : 576<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 17, 17), dtype=float32)</td>
			<td>dim : -3<br>start : 576<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 8, 8), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 8, 8), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 640<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 8, 8), dtype=float32)</td>
			<td>dim : -3<br>start : 640<br>stop : 1024<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 53<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 53<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 53<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 53<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 25<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 25<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 25<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 25<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 11<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 11<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 11<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 11<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=bert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 7<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 1024), dtype=uint1)</td>
			<td>dim : -1<br>start : 0<br>stop : 7<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 4, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 4, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 80), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 80), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 80<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 16<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 16<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 80), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 80), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 80<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 16<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 16<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 2), dtype=float32)</td>
			<td>dim : -2<br>start : 10<br>stop : 11<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 80), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 80), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 80<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 16<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 16<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 3072<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 3072<br>stop : 6144<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 6144<br>stop : 9216<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 96), dtype=float32)</td>
			<td>dim : -1<br>start : 48<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 96), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 48<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 35, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=roberta.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.embed_positions.weights, dtype=float32)</td>
			<td>dim : -2<br>start : 2<br>stop : 258<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 88<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 88<br>stop : 132<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 132<br>stop : 176<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 44<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 44<br>stop : 88<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 88<br>stop : 176<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 64<br>stop : 192<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 192<br>stop : 448<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 384<br>stop : 448<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 128<br>stop : 192<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 192<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 48<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 48<br>stop : 144<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 144<br>stop : 336<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 192<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 192<br>stop : 288<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 288<br>stop : 336<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 96<br>stop : 144<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 144<br>stop : 192<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 48<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 48<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 96<br>stop : 192<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 16<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 16<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 64<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 96<br>stop : 112<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 32<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 96<br>stop : 224<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 128<br>stop : 192<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 192<br>stop : 224<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 64<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 96<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 40<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 40<br>stop : 120<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 120<br>stop : 280<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 160<br>stop : 240<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 240<br>stop : 280<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 80<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 80<br>stop : 120<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 120<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 40<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 40<br>stop : 80<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 80<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 308, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 44<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 308, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 44<br>stop : 132<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 308, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 132<br>stop : 308<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 308, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 176<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 308, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 176<br>stop : 264<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 308, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 264<br>stop : 308<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 210, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 30<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 210, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 30<br>stop : 90<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 210, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 90<br>stop : 210<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 210, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 120<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 210, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 120<br>stop : 180<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 210, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 180<br>stop : 210<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 60<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 60<br>stop : 90<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 90<br>stop : 120<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 30<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 30<br>stop : 60<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 60<br>stop : 120<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 9<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 6<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 9<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 2048), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 1536), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Constant, name=clip_model.text_model.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 7<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 588, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 588, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 577<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 204, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 3, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 3, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 3, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 73, 64), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : -2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 73, 64), dtype=float32)</td>
			<td>dim : -2<br>start : -2<br>stop : -1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 73, 64), dtype=float32)</td>
			<td>dim : -2<br>start : 72<br>stop : 73<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 71, 6, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 71, 6, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 6, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 6, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 3, 64), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 3, 64), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 3, 64), dtype=float32)</td>
			<td>dim : -2<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 16<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 16<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 107, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 107, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 107, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 107, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 107, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 107, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 1024), dtype=uint1)</td>
			<td>dim : -1<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 4, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 4, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 32, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 32, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 32, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 32, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 2048), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 2048), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 512), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 512), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 3072<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 3072<br>stop : 6144<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 6144<br>stop : 9216<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 5, 96), dtype=float32)</td>
			<td>dim : -1<br>start : 48<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 5, 96), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 48<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 2), dtype=float32)</td>
			<td>dim : -2<br>start : 4<br>stop : 5<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 3072<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 3072<br>stop : 6144<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 9216), dtype=float32)</td>
			<td>dim : -1<br>start : 6144<br>stop : 9216<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 13, 96), dtype=float32)</td>
			<td>dim : -1<br>start : 48<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 13, 96), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 48<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 39, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 39, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 39, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 39, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 13, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 13, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 13, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 13, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 29, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_generic_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 96), dtype=float32)</td>
			<td>dim : -1<br>start : -24<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_generic_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 96), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 8), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 4<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 8), dtype=float32)</td>
			<td>dim : -1<br>start : 4<br>stop : 8<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 48), dtype=float32)</td>
			<td>dim : -1<br>start : 12<br>stop : 24<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 48), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 12<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 48), dtype=float32)</td>
			<td>dim : -1<br>start : 36<br>stop : 48<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 48), dtype=float32)</td>
			<td>dim : -1<br>start : 24<br>stop : 36<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(732, 12), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 729<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(732, 12), dtype=float32)</td>
			<td>dim : -2<br>start : 729<br>stop : 732<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 197<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(732, 16), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 729<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(732, 16), dtype=float32)</td>
			<td>dim : -2<br>start : 729<br>stop : 732<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 197<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 30, 40), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 30, 40), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 60, 80), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 60, 80), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 120, 160), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 120, 160), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)</td>
			<td>dim : -3<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 64<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 160<br>stop : 176<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 288<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 304, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 192<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 304, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 192<br>stop : 288<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 304, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 288<br>stop : 304<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 296, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 296, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 160<br>stop : 272<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 296, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 272<br>stop : 296<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 280<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 112<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 112<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 288<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 416<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 416<br>stop : 448<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 416<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 416<br>stop : 448<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 624, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 624, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 384<br>stop : 576<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 624, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 576<br>stop : 624<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 512<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 512<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 768<br>stop : 1024<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 1024<br>stop : 1280<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 1280<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 1536<br>stop : 1792<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 4, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 4, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 4, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -3<br>start : 53<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 53<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -2<br>start : 53<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 53<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 56, 128), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 56, 128), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 8, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 8, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 8, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -3<br>start : 25<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 25<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -2<br>start : 25<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 25<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 28, 256), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 28, 256), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 16, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 16, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 16, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -3<br>start : 11<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 11<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -2<br>start : 11<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 11<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 14, 512), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 14, 512), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 32, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 32, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 32, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
	</tbody>
</table>
