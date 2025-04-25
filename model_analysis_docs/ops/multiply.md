<h1>Comprehensive Report on Multiply Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of multiply operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Multiply Operation Details</th>
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
			<td rowspan="33">1</td>
			<td rowspan="33">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="33">101</td>
			<td>22</td>
			<td><ul><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_50, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_100, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_80, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_30, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_4115, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5153, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_33153, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 1, 1, 13), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_90, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_80, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_30, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_50, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_100, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_40, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_70, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_20, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Constant, name=const_50, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 1, 7, 7), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 201), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 204), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_4115, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 384, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_110, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_60, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="19">2</td>
			<td rowspan="19">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="19">60</td>
			<td>7</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_78680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_149680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_169680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_218680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_157358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.1.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.1.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_42208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_42358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_57358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_102358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_117358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_132358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_1445, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_195452, dtype=float32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
