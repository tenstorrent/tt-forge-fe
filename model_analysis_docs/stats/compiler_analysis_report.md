<h1>Compiler Component Failure Analysis by Model Impacts</h1>
<p>The table highlights the failures encountered in different compiler components, the number of models impacted by each failure, and the specific models affected.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th>Compiler Component</th>
			<th>Failure</th>
			<th>Number of Impacted Models</th>
			<th>Impacted Models</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td rowspan="70">Forge-Fe</td>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: resize2d</td>
			<td>64</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_fpn_base_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_320x320</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_yolox_yolox_nano_obj_det_torchhub</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolox_yolox_darknet_obj_det_torchhub</li><li>pt_yolox_yolox_l_obj_det_torchhub</li><li>pt_yolox_yolox_tiny_obj_det_torchhub</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_yolox_yolox_x_obj_det_torchhub</li><li>pt_yolox_yolox_m_obj_det_torchhub</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li><li>pt_yolox_yolox_s_obj_det_torchhub</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td>58</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 0 - data type mismatch: expected UInt8, got Float32</td>
			<td>47</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td>39</td>
			<td><ul><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td>10</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype Unsupported] RuntimeError Unhandled dtype Bool</td>
			<td>8</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: conv2d_transpose</td>
			<td>7</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_autoencoder_conv_img_enc_github</li><li>pt_yolo_v6_yolov6n_obj_det_torchhub</li><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li><li>pt_yolo_v6_yolov6s_obj_det_torchhub</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1152, 384]</td>
			<td>5</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [1280, 1], got [1, 1001]</td>
			<td>3</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [1225, 1], got [0, 0]</td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [289, 1], got [0, 0]</td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [64, 1], got [0, 0]</td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [16, 1, 1, 1], got [1, 96, 1536, 1536]</td>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 24, 2304, 2304]</td>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 144, 3456, 3456]</td>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 24, 3456, 3456]</td>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 48, 1152, 1152]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 96, 4608, 4608]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 96, 9216, 9216]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 192, 18432, 18432]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 192, 36864, 36864]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 384, 73728, 73728]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 384, 147456, 147456]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 768, 294912, 294912]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [768, 1, 1, 1], got [1, 768, 589824, 589824]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 16, 384, 384]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 48, 6912, 6912]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 288, 13824, 13824]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [288, 1, 1, 1], got [1, 48, 13824, 13824]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [288, 1, 1, 1], got [1, 72, 20736, 20736]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [72, 1, 1, 1], got [1, 432, 31104, 31104]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [432, 1, 1, 1], got [1, 72, 31104, 31104]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [432, 1, 1, 1], got [1, 120, 51840, 51840]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [120, 1, 1, 1], got [1, 720, 86400, 86400]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [720, 1, 1, 1], got [1, 120, 86400, 86400]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [720, 1, 1, 1], got [1, 240, 172800, 172800]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [240, 1, 1, 1], got [1, 1280, 307200, 307200]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 64, 12288, 12288]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [64, 1, 1, 1], got [1, 384, 24576, 24576]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 64, 24576, 24576]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 96, 36864, 36864]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 576, 55296, 55296]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [576, 1, 1, 1], got [1, 96, 55296, 55296]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [576, 1, 1, 1], got [1, 160, 92160, 92160]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [160, 1, 1, 1], got [1, 960, 153600, 153600]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [960, 1, 1, 1], got [1, 160, 153600, 153600]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [960, 1, 1, 1], got [1, 320, 307200, 307200]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [320, 1, 1, 1], got [1, 256, 81920, 81920]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [256, 1, 1, 1], got [1, 21, 5376, 5376]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [16, 1, 1, 1], got [1, 8, 128, 128]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [8, 1, 1, 1], got [1, 48, 384, 384]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 8, 384, 384]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 16, 768, 768]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 16, 1536, 1536]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 32, 4608, 4608]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [32, 1, 1, 1], got [1, 192, 6144, 6144]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 32, 6144, 6144]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 56, 10752, 10752]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [56, 1, 1, 1], got [1, 336, 18816, 18816]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [336, 1, 1, 1], got [1, 56, 18816, 18816]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [336, 1, 1, 1], got [1, 112, 37632, 37632]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [112, 1, 1, 1], got [1, 1280, 143360, 143360]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1], got [1, 12]</td>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [72, 1], got [1, 12]</td>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2304, 768]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 864, 288]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1296, 432]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2160, 720]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [768, 1], got [1, 1001]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [1024, 1], got [1, 1001]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td rowspan="3">MLIR</td>
			<td>[MLIR][MLIR runtime ttnn ] tt::exception tt-mlir/runtime/lib/ttnn/runtime.cpp Unsupported data type</td>
			<td>12</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
		</tr>
		<tr>
			<td>[MLIR][TTIR to TTNN Conv2dOpConversionPattern] tt_forge_signal_handler tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &) adaptor.getPaddingBottom() == adaptor.getPaddingTop() TTNN only supports padding height/width attributes. Thus, padding_top must equal padding_bottom for the op to execute as expected</td>
			<td>6</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[MLIR][mlir::AffineMap collapsedLinearAffineMap] python: /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/lib/Dialect/TT/IR/TTOpsTypes.cpp:460: mlir::AffineMap collapsedLinearAffineMap(::mlir::MLIRContext *, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<std::pair<std::int64_t, std::int64_t>>): Assertion `found && "Dim does not participate in AffineMap RHS"' failed.</td>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
		</tr>
		<tr>
			<td rowspan="10">Metalium</td>
			<td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td>161</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>DeepSeekWrapper_decoder</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_vgg_19_obj_det_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
			<td>32</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolox_yolox_darknet_obj_det_torchhub</li><li>pt_yolox_yolox_l_obj_det_torchhub</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li><li>pt_yolo_v6_yolov6s_obj_det_torchhub</li><li>pt_yolox_yolox_m_obj_det_torchhub</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_yolo_v6_yolov6n_obj_det_torchhub</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
			<td>30</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast</td>
			<td>11</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn.reshape] RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
			<td>10</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][tt-metal kernel] RuntimeError tt-metal/tt_metal/impl/kernels/kernel.cpp unique+common runtime args targeting kernel are too large</td>
			<td>6</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][tt-metal buffer allocation] RuntimeError tt_metal/impl/allocator/bank_manager.cpp Out of Memory: Not enough space to allocate DRAM buffer</td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn conv2d] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp act_block_w_datums == round_up(conv_act_size_c * filter_w, TILE_WIDTH)</td>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn softmax] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B Inputs must be of bfloat16 or bfloat8_b type</td>
			<td>2</td>
			<td><ul><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn shared operation] RuntimeError ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp (*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal.get_alignment(HalMemType::L1) == 0 Shard page size must currently have L1 aligned page size</td>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
		</tr>
	</tbody>
</table>
<h1>Compiler-Specific Model Statistics</h1>
<p>The table summarizes model performance across three compiler components: Forge-Fe, MLIR, and Metalium. It highlights the count of models that passed for each component, along with their respective pass rates, average pass rates and the top 10 models with the lowest pass rates.</p>
<ul><li><b>Models Passed: </b>The count of models that achieved a 100% success rate for a specific compiler component.</li><li><b>Pass Rate (%): </b>The percentage of models that successfully passed a specific compiler component, calculated as (Models Passed / Total Number of Models) × 100</li><li><b>Average Pass Rate (%): </b>The mean pass rate for a compiler component, determined by dividing the sum of pass rates of individual models by the total number of models.</li><li><b>Top-10 Blocked Models (pass rate in %): </b>A list of the 10 models with the lowest pass rates for a specific compiler component, including their exact pass rate percentages.</li></ul>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Total no of models : 297</th>
		</tr>
		<tr style="text-align: center;">
			<th>Compiler Component</th>
			<th>Models Passed</th>
			<th>Pass Rate (%)</th>
			<th>Average Pass Rate (%)</th>
			<th>Top-10 Blocked Models (pass rate in %)</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>Forge-Fe</td>
			<td>125</td>
			<td>42 %</td>
			<td>97 %</td>
			<td><ul><li>pt_opt_facebook_opt_125m_seq_cls_hf (81 %)</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf (81 %)</li><li>pt_opt_facebook_opt_350m_seq_cls_hf (82 %)</li><li>pt_opt_facebook_opt_125m_clm_hf (83 %)</li><li>pt_opt_facebook_opt_125m_qa_hf (83 %)</li><li>pt_opt_facebook_opt_1_3b_clm_hf (83 %)</li><li>pt_opt_facebook_opt_1_3b_qa_hf (83 %)</li><li>pt_opt_facebook_opt_350m_clm_hf (84 %)</li><li>pt_opt_facebook_opt_350m_qa_hf (84 %)</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision (85 %)</li></ul></td>
		</tr>
		<tr>
			<td>MLIR</td>
			<td>125</td>
			<td>42 %</td>
			<td>97 %</td>
			<td><ul><li>pt_opt_facebook_opt_125m_seq_cls_hf (81 %)</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf (81 %)</li><li>pt_opt_facebook_opt_350m_seq_cls_hf (82 %)</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf (83 %)</li><li>pt_opt_facebook_opt_125m_clm_hf (83 %)</li><li>pt_opt_facebook_opt_125m_qa_hf (83 %)</li><li>pt_opt_facebook_opt_1_3b_clm_hf (83 %)</li><li>pt_opt_facebook_opt_1_3b_qa_hf (83 %)</li><li>pt_distilbert_distilbert_base_cased_mlm_hf (84 %)</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf (84 %)</li></ul></td>
		</tr>
		<tr>
			<td>Metalium</td>
			<td>43</td>
			<td>14 %</td>
			<td>94 %</td>
			<td><ul><li>pt_opt_facebook_opt_125m_seq_cls_hf (71 %)</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf (71 %)</li><li>pt_opt_facebook_opt_125m_clm_hf (73 %)</li><li>pt_opt_facebook_opt_1_3b_clm_hf (73 %)</li><li>pt_opt_facebook_opt_350m_seq_cls_hf (73 %)</li><li>pt_opt_facebook_opt_125m_qa_hf (74 %)</li><li>pt_opt_facebook_opt_1_3b_qa_hf (74 %)</li><li>pt_opt_facebook_opt_350m_clm_hf (75 %)</li><li>pt_opt_facebook_opt_350m_qa_hf (76 %)</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf (77 %)</li></ul></td>
		</tr>
	</tbody>
</table>
<h1>Ops-Specific Failure Statistics</h1>
<p>This table provides detailed insights into operation specific statistics, highlighting the number of failed models for each operation and the associated models that encountered issues. Click on an Operation name to view its detailed analysis</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th>ID</th>
			<th>Operation Name</th>
			<th>Number of failed models</th>
			<th>Failed Models</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>1</td>
			<td><a href="../ops/embedding.md">Embedding</a></td>
			<td>101</td>
			<td><ul><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>2</td>
			<td><a href="../ops/conv2d.md">Conv2d</a></td>
			<td>72</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b32_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_l32_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_s16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_s32_224_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li><li>pt_yolo_v6_yolov6n_obj_det_torchhub</li><li>pt_yolo_v6_yolov6s_obj_det_torchhub</li><li>pt_yolox_yolox_darknet_obj_det_torchhub</li><li>pt_yolox_yolox_l_obj_det_torchhub</li><li>pt_yolox_yolox_m_obj_det_torchhub</li><li>pt_yolox_yolox_x_obj_det_torchhub</li></ul></td>
		</tr>
		<tr>
			<td>3</td>
			<td><a href="../ops/cast.md">Cast</a></td>
			<td>69</td>
			<td><ul><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>4</td>
			<td><a href="../ops/resize2d.md">Resize2d</a></td>
			<td>64</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li><li>pt_yolox_yolox_darknet_obj_det_torchhub</li><li>pt_yolox_yolox_l_obj_det_torchhub</li><li>pt_yolox_yolox_m_obj_det_torchhub</li><li>pt_yolox_yolox_nano_obj_det_torchhub</li><li>pt_yolox_yolox_s_obj_det_torchhub</li><li>pt_yolox_yolox_tiny_obj_det_torchhub</li><li>pt_yolox_yolox_x_obj_det_torchhub</li></ul></td>
		</tr>
		<tr>
			<td>5</td>
			<td><a href="../ops/matmul.md">Matmul</a></td>
			<td>55</td>
			<td><ul><li>DeepSeekWrapper_decoder</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_nbeats_seasionality_basis_clm_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg19_obj_det_osmr</li></ul></td>
		</tr>
		<tr>
			<td>6</td>
			<td><a href="../ops/index.md">Index</a></td>
			<td>50</td>
			<td><ul><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>7</td>
			<td><a href="../ops/reshape.md">Reshape</a></td>
			<td>42</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>8</td>
			<td><a href="../ops/repeatinterleave.md">RepeatInterleave</a></td>
			<td>40</td>
			<td><ul><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>9</td>
			<td><a href="../ops/unsqueeze.md">Unsqueeze</a></td>
			<td>40</td>
			<td><ul><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>10</td>
			<td><a href="../ops/add.md">Add</a></td>
			<td>37</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
		</tr>
		<tr>
			<td>11</td>
			<td><a href="../ops/greater.md">Greater</a></td>
			<td>28</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>12</td>
			<td><a href="../ops/multiply.md">Multiply</a></td>
			<td>25</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
		</tr>
		<tr>
			<td>13</td>
			<td><a href="../ops/where.md">Where</a></td>
			<td>17</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
		</tr>
		<tr>
			<td>14</td>
			<td><a href="../ops/maxpool2d.md">MaxPool2d</a></td>
			<td>12</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li><li>pt_autoencoder_conv_img_enc_github</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_xception_xception_img_cls_timm</li></ul></td>
		</tr>
		<tr>
			<td>15</td>
			<td><a href="../ops/cumsum.md">CumSum</a></td>
			<td>11</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
		</tr>
		<tr>
			<td>16</td>
			<td><a href="../ops/max.md">Max</a></td>
			<td>11</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
		</tr>
		<tr>
			<td>17</td>
			<td><a href="../ops/pad.md">Pad</a></td>
			<td>10</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li></ul></td>
		</tr>
		<tr>
			<td>18</td>
			<td><a href="../ops/subtract.md">Subtract</a></td>
			<td>9</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>19</td>
			<td><a href="../ops/conv2dtranspose.md">Conv2dTranspose</a></td>
			<td>7</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li><li>pt_yolo_v6_yolov6n_obj_det_torchhub</li><li>pt_yolo_v6_yolov6s_obj_det_torchhub</li></ul></td>
		</tr>
		<tr>
			<td>20</td>
			<td><a href="../ops/broadcast.md">Broadcast</a></td>
			<td>6</td>
			<td><ul><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li></ul></td>
		</tr>
		<tr>
			<td>21</td>
			<td><a href="../ops/less.md">Less</a></td>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li></ul></td>
		</tr>
		<tr>
			<td>22</td>
			<td><a href="../ops/reduceavg.md">ReduceAvg</a></td>
			<td>6</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
		</tr>
		<tr>
			<td>23</td>
			<td><a href="../ops/transpose.md">Transpose</a></td>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>24</td>
			<td><a href="../ops/softmax.md">Softmax</a></td>
			<td>4</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li></ul></td>
		</tr>
		<tr>
			<td>25</td>
			<td><a href="../ops/argmax.md">Argmax</a></td>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>26</td>
			<td><a href="../ops/equal.md">Equal</a></td>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>27</td>
			<td><a href="../ops/layernorm.md">Layernorm</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>28</td>
			<td><a href="../ops/avgpool2d.md">AvgPool2d</a></td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_v4_img_cls_timm</li></ul></td>
		</tr>
		<tr>
			<td>29</td>
			<td><a href="../ops/concatenate.md">Concatenate</a></td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_v4_img_cls_timm</li></ul></td>
		</tr>
		<tr>
			<td>30</td>
			<td><a href="../ops/notequal.md">NotEqual</a></td>
			<td>2</td>
			<td><ul><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
		</tr>
		<tr>
			<td>31</td>
			<td><a href="../ops/advindex.md">AdvIndex</a></td>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>32</td>
			<td><a href="../ops/avgpool1d.md">AvgPool1d</a></td>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
		</tr>
		<tr>
			<td>33</td>
			<td><a href="../ops/avgpool3d.md">AvgPool3d</a></td>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
		</tr>
	</tbody>
</table>
