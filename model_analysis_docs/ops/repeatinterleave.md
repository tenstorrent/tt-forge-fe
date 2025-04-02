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
			<td rowspan="81">1</td>
			<td rowspan="81">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="81">281</td>
			<td>28</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>28</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 2</td>
		</tr>
		<tr>
			<td>23</td>
			<td><ul><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 2</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 2</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=int64)</td>
			<td>repeats : 256<br>dim : 2</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 2</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 32), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 32), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 32), dtype=int64)</td>
			<td>repeats : 32<br>dim : 2</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 4, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 256, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 256, 128), dtype=float32)</td>
			<td>repeats : 4<br>dim : 2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 4, 128), dtype=float32)</td>
			<td>repeats : 4<br>dim : 2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 35, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1, 1, 768), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1, 13), dtype=int64)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1, 13), dtype=int64)</td>
			<td>repeats : 1<br>dim : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 10, 256), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 1), dtype=float32)</td>
			<td>repeats : 1<br>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 10, 256), dtype=float32)</td>
			<td>repeats : 3<br>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 256, 64), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 256, 64), dtype=float32)</td>
			<td>repeats : 4<br>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 4, 64), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 4, 64), dtype=float32)</td>
			<td>repeats : 4<br>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 35, 128), dtype=float32)</td>
			<td>repeats : 6<br>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 35, 128), dtype=float32)</td>
			<td>repeats : 8<br>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 35, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 35, 128), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 39, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 29, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1, 1, 1024), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
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
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9), dtype=int64)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 10, 256), dtype=float32)</td>
			<td>repeats : 2<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 207, 256), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 207, 256), dtype=float32)</td>
			<td>repeats : 2<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 7, 256), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 7, 256), dtype=float32)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 7, 256), dtype=float32)</td>
			<td>repeats : 8<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 207, 256), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 207, 256), dtype=float32)</td>
			<td>repeats : 2<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 107, 256), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 107, 256), dtype=float32)</td>
			<td>repeats : 1<br>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 107, 256), dtype=float32)</td>
			<td>repeats : 8<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 4, 128), dtype=float32)</td>
			<td>repeats : 3<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 32, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 32, 128), dtype=float32)</td>
			<td>repeats : 3<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 256), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 768), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1280), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 35, 64), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 35, 64), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 39, 128), dtype=float32)</td>
			<td>repeats : 8<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 39, 128), dtype=float32)</td>
			<td>repeats : 6<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 29, 128), dtype=float32)</td>
			<td>repeats : 8<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 29, 128), dtype=float32)</td>
			<td>repeats : 6<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 39, 64), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 39, 64), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 39, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 39, 128), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 29, 64), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1, 29, 64), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 13, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 13, 128), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 29, 128), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 29, 128), dtype=float32)</td>
			<td>repeats : 7<br>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1, 1, 384), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1, 1, 192), dtype=float32)</td>
			<td>repeats : 1<br>dim : 0</td>
		</tr>
	</tbody>
</table>
