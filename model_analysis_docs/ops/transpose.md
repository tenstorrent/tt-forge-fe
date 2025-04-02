<h1>Comprehensive Report on Transpose Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of transpose operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Transpose Operation Details</th>
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
			<td rowspan="1090">1</td>
			<td rowspan="1090">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="1090">2927</td>
			<td>47</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>47</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>45</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>44</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>34</td>
			<td><ul><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>32</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>28</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>24</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>24</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_nbeats_seasionality_basis_clm_hf</li><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 128, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 128, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 128, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(8192, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 8192), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 25088), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_mlp_mixer_base_img_cls_github</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(320, 320), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1280, 320), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(320, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 196), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 16384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 2, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 64, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 4096, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 64, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 5, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 320), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 5, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 64, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 1024, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 320), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 32, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 64, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(30000, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2560, 2560), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(10240, 2560), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2560, 10240), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(14336, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 14336), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1536, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 4), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 8192), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 35), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 128, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 128, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1536, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(16384, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 16384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(9216, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 2560), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 32, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 8, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 128, 4), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_nbeats_trend_basis_clm_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3584, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(18944, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3584, 18944), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 196), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(196, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 39), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(11008, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 11008), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 128, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 128, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 128, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 64, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 128, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128256, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8192, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 2, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(8960, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1536, 8960), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(151936, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(152064, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(11008, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 11008), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(151936, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 197, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 197, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 197, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 197), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 196), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(192, 192), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 192), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(192, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 16, 512), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 16, 16), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 13, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(13, 13, 12), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 13, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 12, 13, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 12, 13, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 64, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1280, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(51200, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 4, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 2304), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 256, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 32, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 32, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 64, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 8, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 512, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 512, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 512, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 128, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1024), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2816, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 2816), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(896, 896), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 896), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4864, 896), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(896, 4864), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(151936, 896), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 29), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024, 72), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1001, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 320), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 8, 7, 96), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 49, 3), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 3, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 3, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 32, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(96, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 8, 7, 7, 96), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(96, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(192, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 7, 4, 7, 192), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 49, 6), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 49, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 32, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 4, 7, 7, 192), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 7, 2, 7, 384), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 49, 12), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 49, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 12, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 12, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 32, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 2, 7, 7, 384), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 49, 24), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 49, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 32, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 384, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 384, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 384, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(9, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(30522, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1536, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(6144, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1536, 6144), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 6, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 64, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 20, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 64, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(5120, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1280, 5120), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 8, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 64, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li></ul></td>
			<td>Operand(type=Parameter, shape=(1, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 6), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 12, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 10, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 256, 10), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(131072, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 207), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 8, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(16384, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 16384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256000, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 107), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 7, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 7, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 7, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 7), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(50257, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128256, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(8192, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 32, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 8, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 64, 4), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 32, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 64, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 80), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 80), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 80), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 80), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 80, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(51200, 2560), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 32, 80), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 11), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 11, 80), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 80), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 80), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 80, 11), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 32, 80), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 12), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 12, 80), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 80), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 80), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 80, 12), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 96, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32064, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 29), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(151936, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 12, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 35, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 35, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 35, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 128, 35), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 28, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 4, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 35, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 35, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 35, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 128, 35), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 16, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 35, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 35, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 35, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 35), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 2, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 2, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 12), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 61, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 61, 12), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 61, 61), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 61, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 61, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 61), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32128, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32128, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 16), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 61, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 61, 16), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 61, 61), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 61, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 61, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 61), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32128, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 9216), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 197, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 197, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 197, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 197), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1920), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1792), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 196), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(196, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 2, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 32, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 4096, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 64, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 64, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(160, 160), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 5, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 160), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 5, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 32, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 1024, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(640, 160), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 640), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(160, 640), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 160), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 32, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 32, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 96, 56), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(288, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 3, 3, 32), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 64, 3, 32), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 49, 3, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(576, 192), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 3, 6, 32), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 16, 6, 32), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 49, 6, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1152, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 3, 12, 32), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 4, 12, 32), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 49, 12, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 1, 7, 768), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 3, 24, 32), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 1, 24, 32), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 49, 24, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 7, 7, 768), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 6, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 6, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 6, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 6), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 32, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 64, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 32, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 32, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 13, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 64, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 64, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 16, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 13, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 64, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 24, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 64, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 24, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 24, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 13, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 64, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 6, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1500, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 64, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(51865, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 20, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 64, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1500, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(51865, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1500, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(51865, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1500, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(51865, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 8, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1500, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1500, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 64, 1500), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(51865, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 20, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 2, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 2, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 2, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 64, 2), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(51866, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7, 8, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 7, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 7), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 8, 7, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 588, 16, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 588), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 588, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 588, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 588, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 588), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(5504, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 5504), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32256, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 32, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 39, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 128, 39), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(102400, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 576), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 577, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 577, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 577), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 577, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596, 32, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 596), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 596, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 596, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 128, 596), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32064, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 204, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 204, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 204, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 204, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 204), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 201, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 201, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 201, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 201), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3129, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 14, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 14, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 14, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 14), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 9, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 9, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 9, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 9), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4608, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 96, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(250880, 1536), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 12, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 384, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 384, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 384, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 64, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(119547, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(9, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(28996, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 8, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 10, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 10), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(131072, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(18176, 4544), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4544, 18176), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4672, 4544), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 71, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 71, 6, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4544, 4544), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(65024, 4544), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 9216), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(23040, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 23040), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(12288, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 334), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 334, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 64, 334), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 16, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 207, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 207, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 207, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 207), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3584, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(14336, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3584, 14336), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256000, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 8, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 7), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 1, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 7, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 7), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 2304), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 2304), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 4, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 207, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 207), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(9216, 2304), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304, 9216), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256000, 2304), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 8, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 1, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 107, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 107, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 107, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 107), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 16, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 107, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 107, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 107, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 107), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(24576, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 24576), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256000, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 256, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 256, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 256, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 128, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(50257, 2560), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 32, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 20, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 32, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 128, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(50257, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 24, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 4, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 4, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 4, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 128, 4), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 24, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 8, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 32, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 128, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128256, 3072), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 32, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 128, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(32000, 4096), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(50272, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(50272, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(50272, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 224, 224), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 256, 224), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 1, 512), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 50176, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 50176), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(322, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 55, 55), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 55, 64, 55), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(322, 322), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3025, 322), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3025, 1, 322), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 3025, 322), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 322, 3025), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 322), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(261, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 3, 224), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(261, 261), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 1, 261), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 50176, 261), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 261, 50176), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 261), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 8, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 2048, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1280, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 8, 160), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 2048, 160), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 160, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 160), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 160), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 160), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 160, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 96, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 2048, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(262, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 32, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 5), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 5, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 5, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 5, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 96, 5), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 32, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 13, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 13, 96), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 13, 96), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 96, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 29, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 29), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 16, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 64, 6), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 14, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 35), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 2, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 35, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 64, 35), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 16, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 39, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 39, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 39, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 39), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 16, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 29, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 29), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 12, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 29, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 29, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 128, 29), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 12, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 39, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 39, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 39, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 128, 39), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 14, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 2, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 39, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 39, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 39, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 64, 39), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 28, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 4, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 39, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 39, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 39, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 128, 39), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 14, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 2, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 29, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 64, 29), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 28, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 4, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 13, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 13, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 13, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 128, 13), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 3584), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 28, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 4, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 29, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 29, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 29, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 128, 29), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(250002, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 8), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 8, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 61, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 61, 8), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 61, 61), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 61, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 61, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 64, 61), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 6), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 1), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 6, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 61, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 61, 6), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 61, 61), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 61, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 61, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 64, 61), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256008, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256008, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_generic_basis_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 72), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_generic_basis_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(96, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 72), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(8, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 72), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(48, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 784), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(12, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(3, 12), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(12, 3), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 12), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(784, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 27, 27, 12), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 27, 27), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 27, 27), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 27, 12, 27), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 197, 12), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 197, 197), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 27, 27, 16), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 27, 27), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 27, 27), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 27, 16, 27), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 197, 16), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 197, 197), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 196), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 6, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 197, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 197, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 197, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 64, 197), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 384), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 196), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 3, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 197, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 197, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 197, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 197), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 192), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 2208), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(18, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1664), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 19200), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 300), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 1, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 300, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 19200), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 160, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 160, 120), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4800), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 2, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 300), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 2, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 300, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 300, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 64, 300), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 4800, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 4800), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 80, 128), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 80, 60), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 1200), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 5, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 320), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 300), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 5, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 300, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 300, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 64, 300), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 1200, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1200), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 40, 320), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 30), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 300), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 8, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 300, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 300, 64), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 300, 64), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 64, 300), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 2048), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 300), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 15, 20, 512), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 20, 15), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(11221, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(21843, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 16, 16, 16, 16), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 3, 16, 16, 16), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 16, 16, 3, 16), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mnist_base_img_cls_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 9216), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mnist_base_img_cls_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(10, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1001, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1001, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 576), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(1280, 960), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Constant, name=classifier.1.weight, dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Constant, name=classifier.4.weight, dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Constant, name=classifier.6.weight, dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1088), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 7392), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 888), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 3712), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 440), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 2520), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1008), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 912), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 672), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 2016), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 784), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 1512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 400), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 3024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 320), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 16, 256), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 160), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 128, 56), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 8, 7, 128), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 128), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 3, 4, 32), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 64, 4, 32), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 49, 4, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 49, 4), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 4, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 4, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 32, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 8, 7, 7, 128), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 7, 4, 7, 256), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 256), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 3, 8, 32), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 16, 8, 32), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 49, 8, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 49, 8), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 49, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 8, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 8, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 32, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 4, 7, 7, 256), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 7, 2, 7, 512), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1536, 512), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 3, 16, 32), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 4, 16, 32), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 49, 16, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 49, 16), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 16, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 16, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 32, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 2, 7, 7, 512), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 1, 7, 1024), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 3, 32, 32), dtype=float32)</td>
			<td>dim0 : -5<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 1, 32, 32), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 49, 32, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 49, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 49, 32), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 49, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 32, 49), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 7, 7, 1024), dtype=float32)</td>
			<td>dim0 : -4<br>dim1 : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 1024), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 3136), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 3, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 6, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 12, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 24, 32), dtype=float32)</td>
			<td>dim0 : -3<br>dim1 : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 85, 1600), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 85, 400), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 85, 100), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
	</tbody>
</table>
