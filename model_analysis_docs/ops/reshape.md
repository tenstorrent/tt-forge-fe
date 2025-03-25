<h1>Comprehensive Report on Reshape Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of reshape operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Reshape Operation Details</th>
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
			<td rowspan="723">1</td>
			<td rowspan="723">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="723">1107</td>
			<td>15</td>
			<td><ul><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>ResNetForImageClassification</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnet_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 1, 1), dtype=float32)</td>
			<td>shape : (1, 2048, 1, 1)</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)</td>
			<td>shape : (1, 1280, 1, 1)</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(32, 1, 3, 3), dtype=float32)</td>
			<td>shape : (32, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 128, 64), dtype=float32)</td>
			<td>shape : (12, 128, 64)</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 128, 128), dtype=float32)</td>
			<td>shape : (1, 12, 128, 128)</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
			<td>shape : (12, 128, 128)</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 128, 64), dtype=float32)</td>
			<td>shape : (1, 12, 128, 64)</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>shape : (144, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>shape : (672, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)</td>
			<td>shape : (1, 768)</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
			<td>shape : (128, 768)</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
			<td>shape : (1, 128, 12, 64)</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 768), dtype=float32)</td>
			<td>shape : (1, 128, 768)</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 128), dtype=float32)</td>
			<td>shape : (12, 64, 128)</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(240, 1, 3, 3), dtype=float32)</td>
			<td>shape : (240, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(480, 1, 3, 3), dtype=float32)</td>
			<td>shape : (480, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>shape : (96, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 1, 3, 3), dtype=float32)</td>
			<td>shape : (64, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>shape : (128, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1), dtype=int64)</td>
			<td>shape : (1, 1)</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)</td>
			<td>shape : (1, 768, 196, 1)</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(144, 1, 5, 5), dtype=float32)</td>
			<td>shape : (144, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>shape : (192, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>shape : (256, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 1, 3, 3), dtype=float32)</td>
			<td>shape : (1024, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 12, 64), dtype=float32)</td>
			<td>shape : (1, 128, 768, 1)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 12, 64), dtype=float32)</td>
			<td>shape : (128, 768)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)</td>
			<td>shape : (1, 256)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 768), dtype=float32)</td>
			<td>shape : (1, 256, 768)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61), dtype=int64)</td>
			<td>shape : (1, 61)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(240, 1, 5, 5), dtype=float32)</td>
			<td>shape : (240, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(480, 1, 5, 5), dtype=float32)</td>
			<td>shape : (480, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1152, 1, 5, 5), dtype=float32)</td>
			<td>shape : (1152, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1152, 1, 3, 3), dtype=float32)</td>
			<td>shape : (1152, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(336, 1, 3, 3), dtype=float32)</td>
			<td>shape : (336, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(16, 1, 3, 3), dtype=float32)</td>
			<td>shape : (16, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(72, 1, 5, 5), dtype=float32)</td>
			<td>shape : (72, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)</td>
			<td>shape : (1, 512, 1, 1)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)</td>
			<td>shape : (256, 1024)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1024), dtype=float32)</td>
			<td>shape : (1, 256, 1024)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 64), dtype=float32)</td>
			<td>shape : (16, 256, 64)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 256), dtype=float32)</td>
			<td>shape : (1, 16, 256, 256)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)</td>
			<td>shape : (16, 256, 256)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 64), dtype=float32)</td>
			<td>shape : (1, 16, 256, 64)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
			<td>shape : (256, 1024)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 768), dtype=float32)</td>
			<td>shape : (256, 768)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 256, 64), dtype=float32)</td>
			<td>shape : (12, 256, 64)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 256, 256), dtype=float32)</td>
			<td>shape : (1, 12, 256, 256)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 256, 256), dtype=float32)</td>
			<td>shape : (12, 256, 256)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 256, 64), dtype=float32)</td>
			<td>shape : (1, 12, 256, 64)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 12, 64), dtype=float32)</td>
			<td>shape : (256, 768)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)</td>
			<td>shape : (197, 768)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)</td>
			<td>shape : (1, 197, 12, 64)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 768), dtype=float32)</td>
			<td>shape : (1, 197, 768)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 197, 64), dtype=float32)</td>
			<td>shape : (12, 197, 64)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 197, 197), dtype=float32)</td>
			<td>shape : (1, 12, 197, 197)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 197, 197), dtype=float32)</td>
			<td>shape : (12, 197, 197)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 197), dtype=float32)</td>
			<td>shape : (12, 64, 197)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 197, 64), dtype=float32)</td>
			<td>shape : (1, 12, 197, 64)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 12, 64), dtype=float32)</td>
			<td>shape : (197, 768)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)</td>
			<td>shape : (1, 1024, 1, 1)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(576, 1, 3, 3), dtype=float32)</td>
			<td>shape : (576, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(8, 1, 3, 3), dtype=float32)</td>
			<td>shape : (8, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(24, 1, 3, 3), dtype=float32)</td>
			<td>shape : (24, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>shape : (48, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(12, 1, 3, 3), dtype=float32)</td>
			<td>shape : (12, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(36, 1, 3, 3), dtype=float32)</td>
			<td>shape : (36, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(20, 1, 3, 3), dtype=float32)</td>
			<td>shape : (20, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(24, 1, 5, 5), dtype=float32)</td>
			<td>shape : (24, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(60, 1, 3, 3), dtype=float32)</td>
			<td>shape : (60, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(120, 1, 3, 3), dtype=float32)</td>
			<td>shape : (120, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(40, 1, 3, 3), dtype=float32)</td>
			<td>shape : (40, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(100, 1, 3, 3), dtype=float32)</td>
			<td>shape : (100, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(92, 1, 3, 3), dtype=float32)</td>
			<td>shape : (92, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(56, 1, 3, 3), dtype=float32)</td>
			<td>shape : (56, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(80, 1, 3, 3), dtype=float32)</td>
			<td>shape : (80, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(112, 1, 5, 5), dtype=float32)</td>
			<td>shape : (112, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)</td>
			<td>shape : (1, 1, 1024)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384), dtype=float32)</td>
			<td>shape : (1, 1, 6, 64)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 64), dtype=float32)</td>
			<td>shape : (6, 1, 64)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 1), dtype=float32)</td>
			<td>shape : (1, 6, 1, 1)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 1), dtype=float32)</td>
			<td>shape : (6, 1, 1)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 64, 1), dtype=float32)</td>
			<td>shape : (6, 64, 1)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 64), dtype=float32)</td>
			<td>shape : (1, 6, 1, 64)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 6, 64), dtype=float32)</td>
			<td>shape : (1, 384)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)</td>
			<td>shape : (1, 256, 16, 64)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7), dtype=int64)</td>
			<td>shape : (1, 7)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 768), dtype=float32)</td>
			<td>shape : (7, 768)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 768), dtype=float32)</td>
			<td>shape : (1, 7, 12, 64)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 768), dtype=float32)</td>
			<td>shape : (1, 7, 768)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 7, 64), dtype=float32)</td>
			<td>shape : (12, 7, 64)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 7, 7), dtype=float32)</td>
			<td>shape : (1, 12, 7, 7)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 7, 7), dtype=float32)</td>
			<td>shape : (12, 7, 7)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 7), dtype=float32)</td>
			<td>shape : (12, 64, 7)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 7, 64), dtype=float32)</td>
			<td>shape : (1, 12, 7, 64)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 12, 64), dtype=float32)</td>
			<td>shape : (7, 768)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 3072), dtype=float32)</td>
			<td>shape : (1, 7, 3072)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 3072), dtype=float32)</td>
			<td>shape : (7, 3072)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 768), dtype=float32)</td>
			<td>shape : (1, 256, 12, 64)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 256), dtype=float32)</td>
			<td>shape : (12, 64, 256)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)</td>
			<td>shape : (1, 512)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)</td>
			<td>shape : (1, 256, 8, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 32), dtype=float32)</td>
			<td>shape : (8, 256, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 256), dtype=float32)</td>
			<td>shape : (1, 256, 256)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 256), dtype=float32)</td>
			<td>shape : (1, 8, 256, 256)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 256), dtype=float32)</td>
			<td>shape : (8, 256, 256)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512), dtype=float32)</td>
			<td>shape : (1, 1, 512)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 512), dtype=float32)</td>
			<td>shape : (61, 512)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 512), dtype=float32)</td>
			<td>shape : (1, 61, 512)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 1024), dtype=float32)</td>
			<td>shape : (1, 61, 1024)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 6, 6), dtype=float32)</td>
			<td>shape : (1, 9216, 1, 1)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)</td>
			<td>shape : (1, 1024)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(288, 1, 5, 5), dtype=float32)</td>
			<td>shape : (288, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(288, 1, 3, 3), dtype=float32)</td>
			<td>shape : (288, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(192, 1, 5, 5), dtype=float32)</td>
			<td>shape : (192, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(672, 1, 3, 3), dtype=float32)</td>
			<td>shape : (672, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>shape : (960, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 1, 3, 3), dtype=float32)</td>
			<td>shape : (512, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>shape : (384, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(960, 1, 3, 3), dtype=float32)</td>
			<td>shape : (960, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>shape : (1, 8, 7, 8, 7, 96)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 8, 7, 7, 96), dtype=float32)</td>
			<td>shape : (3136, 96)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 3, 49, 32), dtype=float32)</td>
			<td>shape : (192, 49, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 49, 49), dtype=float32)</td>
			<td>shape : (64, 3, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2401, 3), dtype=float32)</td>
			<td>shape : (49, 49, 3)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 3, 49, 49), dtype=float32)</td>
			<td>shape : (192, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 3, 49, 49), dtype=float32)</td>
			<td>shape : (1, 64, 3, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 3, 32, 49), dtype=float32)</td>
			<td>shape : (192, 32, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 49, 32), dtype=float32)</td>
			<td>shape : (64, 3, 49, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 3, 32), dtype=float32)</td>
			<td>shape : (3136, 96)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 96), dtype=float32)</td>
			<td>shape : (64, 49, 96)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 96), dtype=float32)</td>
			<td>shape : (1, 8, 8, 7, 7, 96)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 8, 7, 96), dtype=float32)</td>
			<td>shape : (1, 56, 56, 96)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 3, 49, 49), dtype=float32)</td>
			<td>shape : (64, 3, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 192), dtype=float32)</td>
			<td>shape : (16, 49, 192)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>shape : (1, 4, 7, 4, 7, 192)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 4, 7, 7, 192), dtype=float32)</td>
			<td>shape : (784, 192)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 49, 32), dtype=float32)</td>
			<td>shape : (96, 49, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 49, 49), dtype=float32)</td>
			<td>shape : (16, 6, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2401, 6), dtype=float32)</td>
			<td>shape : (49, 49, 6)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 49, 49), dtype=float32)</td>
			<td>shape : (96, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 49, 49), dtype=float32)</td>
			<td>shape : (1, 16, 6, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 32, 49), dtype=float32)</td>
			<td>shape : (96, 32, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 49, 32), dtype=float32)</td>
			<td>shape : (16, 6, 49, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 6, 32), dtype=float32)</td>
			<td>shape : (784, 192)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 192), dtype=float32)</td>
			<td>shape : (1, 4, 4, 7, 7, 192)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 7, 4, 7, 192), dtype=float32)</td>
			<td>shape : (1, 28, 28, 192)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 49, 49), dtype=float32)</td>
			<td>shape : (16, 6, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 384), dtype=float32)</td>
			<td>shape : (4, 49, 384)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>shape : (1, 2, 7, 2, 7, 384)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 2, 7, 7, 384), dtype=float32)</td>
			<td>shape : (196, 384)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 12, 49, 32), dtype=float32)</td>
			<td>shape : (48, 49, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 49, 49), dtype=float32)</td>
			<td>shape : (4, 12, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2401, 12), dtype=float32)</td>
			<td>shape : (49, 49, 12)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 12, 49, 49), dtype=float32)</td>
			<td>shape : (48, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 12, 49, 49), dtype=float32)</td>
			<td>shape : (1, 4, 12, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 12, 32, 49), dtype=float32)</td>
			<td>shape : (48, 32, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 49, 32), dtype=float32)</td>
			<td>shape : (4, 12, 49, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 12, 32), dtype=float32)</td>
			<td>shape : (196, 384)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 384), dtype=float32)</td>
			<td>shape : (1, 2, 2, 7, 7, 384)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 7, 2, 7, 384), dtype=float32)</td>
			<td>shape : (1, 14, 14, 384)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 12, 49, 49), dtype=float32)</td>
			<td>shape : (4, 12, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 768), dtype=float32)</td>
			<td>shape : (1, 49, 768)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 49, 32), dtype=float32)</td>
			<td>shape : (24, 49, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 49, 49), dtype=float32)</td>
			<td>shape : (1, 24, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2401, 24), dtype=float32)</td>
			<td>shape : (49, 49, 24)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 49, 49), dtype=float32)</td>
			<td>shape : (24, 49, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 32, 49), dtype=float32)</td>
			<td>shape : (24, 32, 49)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 49, 32), dtype=float32)</td>
			<td>shape : (1, 24, 49, 32)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 24, 32), dtype=float32)</td>
			<td>shape : (49, 768)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(728, 1, 3, 3), dtype=float32)</td>
			<td>shape : (728, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1536, 1, 3, 3), dtype=float32)</td>
			<td>shape : (1536, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1), dtype=int64)</td>
			<td>shape : (2, 4, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1), dtype=int64)</td>
			<td>shape : (2, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)</td>
			<td>shape : (1, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)</td>
			<td>shape : (1, 1, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1024), dtype=float32)</td>
			<td>shape : (2, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1024), dtype=float32)</td>
			<td>shape : (2, 1, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1024), dtype=float32)</td>
			<td>shape : (2, 1, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1024), dtype=float32)</td>
			<td>shape : (2, 1, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 16, 1, 64), dtype=float32)</td>
			<td>shape : (32, 1, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 64), dtype=float32)</td>
			<td>shape : (2, 16, 1, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 16, 64), dtype=float32)</td>
			<td>shape : (2, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13), dtype=int64)</td>
			<td>shape : (2, 13)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 768), dtype=float32)</td>
			<td>shape : (26, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 768), dtype=float32)</td>
			<td>shape : (2, 13, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 768), dtype=float32)</td>
			<td>shape : (2, 13, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 12, 13, 64), dtype=float32)</td>
			<td>shape : (24, 13, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 13, 13), dtype=float32)</td>
			<td>shape : (2, 12, 13, 13)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 12, 13, 13), dtype=float32)</td>
			<td>shape : (24, 13, 13)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 12, 64, 13), dtype=float32)</td>
			<td>shape : (24, 64, 13)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 13, 64), dtype=float32)</td>
			<td>shape : (2, 12, 13, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 12, 64), dtype=float32)</td>
			<td>shape : (26, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 3072), dtype=float32)</td>
			<td>shape : (2, 13, 3072)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 3072), dtype=float32)</td>
			<td>shape : (26, 3072)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 1024), dtype=float32)</td>
			<td>shape : (2, 13, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 1024), dtype=float32)</td>
			<td>shape : (2, 13, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 1024), dtype=float32)</td>
			<td>shape : (26, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 16, 13, 64), dtype=float32)</td>
			<td>shape : (32, 13, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 13), dtype=float32)</td>
			<td>shape : (2, 16, 1, 13)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 16, 1, 13), dtype=float32)</td>
			<td>shape : (32, 1, 13)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4096), dtype=float32)</td>
			<td>shape : (2, 1, 4096)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 4096), dtype=float32)</td>
			<td>shape : (2, 4096)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 2048), dtype=float32)</td>
			<td>shape : (2, 1, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1, 2048), dtype=float32)</td>
			<td>shape : (8, 1, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 384), dtype=float32)</td>
			<td>shape : (1, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 384), dtype=float32)</td>
			<td>shape : (1, 1, 6, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384), dtype=float32)</td>
			<td>shape : (1, 1, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000), dtype=float32)</td>
			<td>shape : (1, 80, 3000, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 80, 3), dtype=float32)</td>
			<td>shape : (384, 80, 3, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 3000, 1), dtype=float32)</td>
			<td>shape : (1, 384, 3000)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 3000), dtype=float32)</td>
			<td>shape : (1, 384, 3000, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 384, 3), dtype=float32)</td>
			<td>shape : (384, 384, 3, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 1500, 1), dtype=float32)</td>
			<td>shape : (1, 384, 1500)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 384), dtype=float32)</td>
			<td>shape : (1500, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 384), dtype=float32)</td>
			<td>shape : (1, 1500, 6, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1500, 384), dtype=float32)</td>
			<td>shape : (1, 1500, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1500, 384), dtype=float32)</td>
			<td>shape : (1, 1500, 6, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1500, 64), dtype=float32)</td>
			<td>shape : (6, 1500, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1500, 1500), dtype=float32)</td>
			<td>shape : (1, 6, 1500, 1500)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1500, 1500), dtype=float32)</td>
			<td>shape : (6, 1500, 1500)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 64, 1500), dtype=float32)</td>
			<td>shape : (6, 64, 1500)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1500, 64), dtype=float32)</td>
			<td>shape : (1, 6, 1500, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 6, 64), dtype=float32)</td>
			<td>shape : (1500, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 1500), dtype=float32)</td>
			<td>shape : (1, 6, 1, 1500)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 1500), dtype=float32)</td>
			<td>shape : (6, 1, 1500)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7), dtype=int64)</td>
			<td>shape : (2, 7)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7, 512), dtype=float32)</td>
			<td>shape : (14, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7, 512), dtype=float32)</td>
			<td>shape : (2, 7, 8, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 512), dtype=float32)</td>
			<td>shape : (2, 7, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 8, 7, 64), dtype=float32)</td>
			<td>shape : (16, 7, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 7, 7), dtype=float32)</td>
			<td>shape : (2, 8, 7, 7)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 8, 7, 7), dtype=float32)</td>
			<td>shape : (2, 8, 7, 7)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 8, 7, 7), dtype=float32)</td>
			<td>shape : (16, 7, 7)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 7, 64), dtype=float32)</td>
			<td>shape : (2, 8, 7, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7, 8, 64), dtype=float32)</td>
			<td>shape : (14, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 2048), dtype=float32)</td>
			<td>shape : (2, 7, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7, 2048), dtype=float32)</td>
			<td>shape : (14, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 4096), dtype=float32)</td>
			<td>shape : (39, 4096)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 4096), dtype=float32)</td>
			<td>shape : (1, 39, 32, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 4096), dtype=float32)</td>
			<td>shape : (1, 39, 4096)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 128), dtype=float32)</td>
			<td>shape : (32, 39, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 39, 39), dtype=float32)</td>
			<td>shape : (1, 32, 39, 39)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 39), dtype=float32)</td>
			<td>shape : (32, 39, 39)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128, 39), dtype=float32)</td>
			<td>shape : (32, 128, 39)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 39, 128), dtype=float32)</td>
			<td>shape : (1, 32, 39, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 32, 128), dtype=float32)</td>
			<td>shape : (39, 4096)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 11008), dtype=float32)</td>
			<td>shape : (1, 39, 11008)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 768), dtype=float32)</td>
			<td>shape : (201, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 768), dtype=float32)</td>
			<td>shape : (1, 201, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(201, 768), dtype=float32)</td>
			<td>shape : (1, 201, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 201, 64), dtype=float32)</td>
			<td>shape : (12, 201, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 201, 201), dtype=float32)</td>
			<td>shape : (1, 12, 201, 201)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 201, 201), dtype=float32)</td>
			<td>shape : (12, 201, 201)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 201), dtype=float32)</td>
			<td>shape : (12, 64, 201)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 201, 64), dtype=float32)</td>
			<td>shape : (1, 12, 201, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 12, 64), dtype=float32)</td>
			<td>shape : (201, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 128), dtype=float32)</td>
			<td>shape : (1, 768, 128, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 768), dtype=float32)</td>
			<td>shape : (14, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 768), dtype=float32)</td>
			<td>shape : (1, 14, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 768), dtype=float32)</td>
			<td>shape : (1, 14, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 14, 64), dtype=float32)</td>
			<td>shape : (12, 14, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 14, 14), dtype=float32)</td>
			<td>shape : (1, 12, 14, 14)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 14, 14), dtype=float32)</td>
			<td>shape : (12, 14, 14)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 14), dtype=float32)</td>
			<td>shape : (12, 64, 14)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 14, 64), dtype=float32)</td>
			<td>shape : (1, 12, 14, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 12, 64), dtype=float32)</td>
			<td>shape : (1, 14, 768, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 1), dtype=float32)</td>
			<td>shape : (1, 14, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 768), dtype=float32)</td>
			<td>shape : (9, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 768), dtype=float32)</td>
			<td>shape : (1, 9, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(9, 768), dtype=float32)</td>
			<td>shape : (1, 9, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 9, 64), dtype=float32)</td>
			<td>shape : (12, 9, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 9, 9), dtype=float32)</td>
			<td>shape : (1, 12, 9, 9)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 9, 9), dtype=float32)</td>
			<td>shape : (12, 9, 9)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 9), dtype=float32)</td>
			<td>shape : (12, 64, 9)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 9, 64), dtype=float32)</td>
			<td>shape : (1, 12, 9, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 12, 64), dtype=float32)</td>
			<td>shape : (1, 9, 768, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1024), dtype=float32)</td>
			<td>shape : (1, 256, 4, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 1024), dtype=float32)</td>
			<td>shape : (384, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 1024), dtype=float32)</td>
			<td>shape : (1, 384, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 1024), dtype=float32)</td>
			<td>shape : (1, 384, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 384, 64), dtype=float32)</td>
			<td>shape : (16, 384, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 384, 384), dtype=float32)</td>
			<td>shape : (1, 16, 384, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 384, 384), dtype=float32)</td>
			<td>shape : (16, 384, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 64, 384), dtype=float32)</td>
			<td>shape : (16, 64, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 384, 64), dtype=float32)</td>
			<td>shape : (1, 16, 384, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 16, 64), dtype=float32)</td>
			<td>shape : (384, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 1), dtype=float32)</td>
			<td>shape : (1, 384, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 768), dtype=float32)</td>
			<td>shape : (6, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 768), dtype=float32)</td>
			<td>shape : (1, 6, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 768), dtype=float32)</td>
			<td>shape : (1, 6, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 6, 64), dtype=float32)</td>
			<td>shape : (12, 6, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 6, 6), dtype=float32)</td>
			<td>shape : (1, 12, 6, 6)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 6, 6), dtype=float32)</td>
			<td>shape : (12, 6, 6)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 6), dtype=float32)</td>
			<td>shape : (12, 64, 6)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 6, 64), dtype=float32)</td>
			<td>shape : (1, 12, 6, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 12, 64), dtype=float32)</td>
			<td>shape : (6, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4608), dtype=float32)</td>
			<td>shape : (1, 32, 16, 3, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 1, 96), dtype=float32)</td>
			<td>shape : (1, 32, 16, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 96), dtype=float32)</td>
			<td>shape : (16, 32, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32), dtype=float32)</td>
			<td>shape : (16, 1, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 32), dtype=float32)</td>
			<td>shape : (1, 16, 32, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 32), dtype=float32)</td>
			<td>shape : (16, 32, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 96), dtype=float32)</td>
			<td>shape : (1, 16, 32, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 96), dtype=float32)</td>
			<td>shape : (32, 1536)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1536), dtype=float32)</td>
			<td>shape : (1, 32, 1536)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4, 256), dtype=float32)</td>
			<td>shape : (1, 256, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 16, 2), dtype=float32)</td>
			<td>shape : (1, 256, 16, 32, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 64, 256), dtype=float32)</td>
			<td>shape : (16, 64, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 4096), dtype=float32)</td>
			<td>shape : (1, 256, 4096)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=uint1)</td>
			<td>shape : (1, 1, 1, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 2048), dtype=float32)</td>
			<td>shape : (10, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 2048), dtype=float32)</td>
			<td>shape : (1, 10, 8, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 2048), dtype=float32)</td>
			<td>shape : (1, 10, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 256), dtype=float32)</td>
			<td>shape : (8, 10, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 1024), dtype=float32)</td>
			<td>shape : (1, 10, 4, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 2, 10, 256), dtype=float32)</td>
			<td>shape : (8, 10, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 2, 10, 256), dtype=float32)</td>
			<td>shape : (1, 8, 10, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 10, 10), dtype=float32)</td>
			<td>shape : (1, 8, 10, 10)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 10), dtype=float32)</td>
			<td>shape : (8, 10, 10)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 10), dtype=float32)</td>
			<td>shape : (8, 256, 10)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 10, 256), dtype=float32)</td>
			<td>shape : (1, 8, 10, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 8, 256), dtype=float32)</td>
			<td>shape : (10, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 8192), dtype=float32)</td>
			<td>shape : (1, 10, 8192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 12288), dtype=float32)</td>
			<td>shape : (1, 334, 64, 3, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 1, 64), dtype=float32)</td>
			<td>shape : (1, 334, 64, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 64), dtype=float32)</td>
			<td>shape : (64, 334, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 334, 334), dtype=float32)</td>
			<td>shape : (1, 64, 334, 334)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 334), dtype=float32)</td>
			<td>shape : (64, 334, 334)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 64, 334), dtype=float32)</td>
			<td>shape : (64, 64, 334)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 334, 64), dtype=float32)</td>
			<td>shape : (1, 64, 334, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 64), dtype=float32)</td>
			<td>shape : (334, 4096)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(334, 4096), dtype=float32)</td>
			<td>shape : (1, 334, 4096)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 2304), dtype=float32)</td>
			<td>shape : (207, 2304)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 2048), dtype=float32)</td>
			<td>shape : (1, 207, 8, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 256), dtype=float32)</td>
			<td>shape : (8, 207, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 1024), dtype=float32)</td>
			<td>shape : (1, 207, 4, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 2, 207, 256), dtype=float32)</td>
			<td>shape : (8, 207, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 2, 207, 256), dtype=float32)</td>
			<td>shape : (1, 8, 207, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 207, 207), dtype=float32)</td>
			<td>shape : (1, 8, 207, 207)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 207), dtype=float32)</td>
			<td>shape : (8, 207, 207)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 207), dtype=float32)</td>
			<td>shape : (8, 256, 207)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 207, 256), dtype=float32)</td>
			<td>shape : (1, 8, 207, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 8, 256), dtype=float32)</td>
			<td>shape : (207, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 2304), dtype=float32)</td>
			<td>shape : (1, 207, 2304)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 9216), dtype=float32)</td>
			<td>shape : (1, 207, 9216)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 768), dtype=float32)</td>
			<td>shape : (1, 7, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 2), dtype=float32)</td>
			<td>shape : (7, 2)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2), dtype=float32)</td>
			<td>shape : (1, 2)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 768), dtype=float32)</td>
			<td>shape : (1, 256, 8, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 768), dtype=float32)</td>
			<td>shape : (1, 256, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=float32)</td>
			<td>shape : (1, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 3072), dtype=float32)</td>
			<td>shape : (1, 256, 3072)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 3072), dtype=float32)</td>
			<td>shape : (256, 3072)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
			<td>shape : (1, 1, 1, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
			<td>shape : (1, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 322), dtype=float32)</td>
			<td>shape : (1, 512, 1, 322)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 55, 55, 64), dtype=float32)</td>
			<td>shape : (1, 3025, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3025, 322), dtype=float32)</td>
			<td>shape : (3025, 322)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3025, 322), dtype=float32)</td>
			<td>shape : (1, 3025, 1, 322)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3025, 322), dtype=float32)</td>
			<td>shape : (1, 3025, 322)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 3025), dtype=float32)</td>
			<td>shape : (1, 1, 512, 3025)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512, 3025), dtype=float32)</td>
			<td>shape : (1, 512, 3025)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 322, 3025), dtype=float32)</td>
			<td>shape : (1, 322, 3025)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
			<td>shape : (512, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
			<td>shape : (1, 512, 8, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
			<td>shape : (1, 512, 1, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(512, 1024), dtype=float32)</td>
			<td>shape : (1, 512, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 512, 128), dtype=float32)</td>
			<td>shape : (8, 512, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 512, 512), dtype=float32)</td>
			<td>shape : (1, 8, 512, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 512, 512), dtype=float32)</td>
			<td>shape : (8, 512, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 128, 512), dtype=float32)</td>
			<td>shape : (8, 128, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 512, 128), dtype=float32)</td>
			<td>shape : (1, 8, 512, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 8, 128), dtype=float32)</td>
			<td>shape : (512, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)</td>
			<td>shape : (1, 1, 1, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 512), dtype=float32)</td>
			<td>shape : (1, 1, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024, 512), dtype=float32)</td>
			<td>shape : (1, 1024, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1000), dtype=float32)</td>
			<td>shape : (1, 1000)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 256), dtype=float32)</td>
			<td>shape : (1, 2048, 8, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 2048, 32), dtype=float32)</td>
			<td>shape : (8, 2048, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)</td>
			<td>shape : (256, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)</td>
			<td>shape : (1, 256, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 768), dtype=float32)</td>
			<td>shape : (2048, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 256), dtype=float32)</td>
			<td>shape : (1, 2048, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 2048), dtype=float32)</td>
			<td>shape : (1, 8, 256, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 2048), dtype=float32)</td>
			<td>shape : (8, 256, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 1280), dtype=float32)</td>
			<td>shape : (1, 2048, 1280)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 1280), dtype=float32)</td>
			<td>shape : (1, 2048, 8, 160)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 160, 2048), dtype=float32)</td>
			<td>shape : (8, 160, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 160), dtype=float32)</td>
			<td>shape : (1, 8, 256, 160)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 160), dtype=float32)</td>
			<td>shape : (256, 1280)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1280), dtype=float32)</td>
			<td>shape : (1, 256, 1280)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1280), dtype=float32)</td>
			<td>shape : (256, 1280)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1280), dtype=float32)</td>
			<td>shape : (1, 256, 8, 160)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 160, 256), dtype=float32)</td>
			<td>shape : (8, 160, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 2048, 256), dtype=float32)</td>
			<td>shape : (1, 8, 2048, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 2048, 256), dtype=float32)</td>
			<td>shape : (8, 2048, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 96, 256), dtype=float32)</td>
			<td>shape : (8, 96, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 2048, 96), dtype=float32)</td>
			<td>shape : (1, 8, 2048, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 8, 96), dtype=float32)</td>
			<td>shape : (2048, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 768), dtype=float32)</td>
			<td>shape : (1, 2048, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 262), dtype=float32)</td>
			<td>shape : (1, 2048, 262)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1024), dtype=float32)</td>
			<td>shape : (6, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1024), dtype=float32)</td>
			<td>shape : (1, 6, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1024), dtype=float32)</td>
			<td>shape : (1, 6, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 64), dtype=float32)</td>
			<td>shape : (16, 6, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 6), dtype=float32)</td>
			<td>shape : (1, 16, 6, 6)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 6), dtype=float32)</td>
			<td>shape : (16, 6, 6)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 64, 6), dtype=float32)</td>
			<td>shape : (16, 64, 6)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 64), dtype=float32)</td>
			<td>shape : (1, 16, 6, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 16, 64), dtype=float32)</td>
			<td>shape : (6, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 2816), dtype=float32)</td>
			<td>shape : (1, 6, 2816)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 896), dtype=float32)</td>
			<td>shape : (35, 896)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 896), dtype=float32)</td>
			<td>shape : (1, 35, 14, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 896), dtype=float32)</td>
			<td>shape : (1, 35, 896)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 64), dtype=float32)</td>
			<td>shape : (14, 35, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 128), dtype=float32)</td>
			<td>shape : (1, 35, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 128), dtype=float32)</td>
			<td>shape : (1, 35, 2, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 7, 35, 64), dtype=float32)</td>
			<td>shape : (14, 35, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 7, 35, 64), dtype=float32)</td>
			<td>shape : (1, 14, 35, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 35, 35), dtype=float32)</td>
			<td>shape : (1, 14, 35, 35)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 35), dtype=float32)</td>
			<td>shape : (14, 35, 35)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 64, 35), dtype=float32)</td>
			<td>shape : (14, 64, 35)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 35, 64), dtype=float32)</td>
			<td>shape : (1, 14, 35, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 14, 64), dtype=float32)</td>
			<td>shape : (35, 896)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 4864), dtype=float32)</td>
			<td>shape : (1, 35, 4864)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 896), dtype=float32)</td>
			<td>shape : (29, 896)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 896), dtype=float32)</td>
			<td>shape : (1, 29, 14, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 896), dtype=float32)</td>
			<td>shape : (1, 29, 896)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 64), dtype=float32)</td>
			<td>shape : (14, 29, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 128), dtype=float32)</td>
			<td>shape : (1, 29, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 128), dtype=float32)</td>
			<td>shape : (1, 29, 2, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 7, 29, 64), dtype=float32)</td>
			<td>shape : (14, 29, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 7, 29, 64), dtype=float32)</td>
			<td>shape : (1, 14, 29, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 29, 29), dtype=float32)</td>
			<td>shape : (1, 14, 29, 29)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 29), dtype=float32)</td>
			<td>shape : (14, 29, 29)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 64, 29), dtype=float32)</td>
			<td>shape : (14, 64, 29)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 29, 64), dtype=float32)</td>
			<td>shape : (1, 14, 29, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 14, 64), dtype=float32)</td>
			<td>shape : (29, 896)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 4864), dtype=float32)</td>
			<td>shape : (1, 29, 4864)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128), dtype=float32)</td>
			<td>shape : (1, 12, 64, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128), dtype=float32)</td>
			<td>shape : (12, 64, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 768, 1), dtype=float32)</td>
			<td>shape : (768, 768, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128, 1), dtype=float32)</td>
			<td>shape : (1, 768, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512), dtype=float32)</td>
			<td>shape : (1, 1, 8, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 64), dtype=float32)</td>
			<td>shape : (8, 1, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
			<td>shape : (1, 8, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)</td>
			<td>shape : (8, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 64, 1), dtype=float32)</td>
			<td>shape : (8, 64, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 64), dtype=float32)</td>
			<td>shape : (1, 8, 1, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 8, 64), dtype=float32)</td>
			<td>shape : (1, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 512), dtype=float32)</td>
			<td>shape : (1, 61, 8, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 61, 64), dtype=float32)</td>
			<td>shape : (8, 61, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 61, 61), dtype=float32)</td>
			<td>shape : (1, 8, 61, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 61, 61), dtype=float32)</td>
			<td>shape : (8, 61, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 64, 61), dtype=float32)</td>
			<td>shape : (8, 64, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 61, 64), dtype=float32)</td>
			<td>shape : (1, 8, 61, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 8, 64), dtype=float32)</td>
			<td>shape : (61, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 61), dtype=float32)</td>
			<td>shape : (1, 8, 1, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 61), dtype=float32)</td>
			<td>shape : (8, 1, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 64), dtype=float32)</td>
			<td>shape : (16, 1, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
			<td>shape : (1, 16, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)</td>
			<td>shape : (16, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)</td>
			<td>shape : (1, 4, 4)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 64, 1), dtype=float32)</td>
			<td>shape : (16, 64, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 64), dtype=float32)</td>
			<td>shape : (1, 16, 1, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 16, 64), dtype=float32)</td>
			<td>shape : (1, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 1024), dtype=float32)</td>
			<td>shape : (61, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 1024), dtype=float32)</td>
			<td>shape : (1, 61, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 61, 64), dtype=float32)</td>
			<td>shape : (16, 61, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 61, 61), dtype=float32)</td>
			<td>shape : (1, 16, 61, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 61, 61), dtype=float32)</td>
			<td>shape : (16, 61, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 64, 61), dtype=float32)</td>
			<td>shape : (16, 64, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 61, 64), dtype=float32)</td>
			<td>shape : (1, 16, 61, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 16, 64), dtype=float32)</td>
			<td>shape : (61, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 61), dtype=float32)</td>
			<td>shape : (1, 16, 1, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 61), dtype=float32)</td>
			<td>shape : (16, 1, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)</td>
			<td>shape : (1, 1, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)</td>
			<td>shape : (1, 1, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 64), dtype=float32)</td>
			<td>shape : (12, 1, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 1), dtype=float32)</td>
			<td>shape : (1, 12, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 1), dtype=float32)</td>
			<td>shape : (12, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 1), dtype=float32)</td>
			<td>shape : (12, 64, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 64), dtype=float32)</td>
			<td>shape : (1, 12, 1, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 12, 64), dtype=float32)</td>
			<td>shape : (1, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 768), dtype=float32)</td>
			<td>shape : (61, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 768), dtype=float32)</td>
			<td>shape : (1, 61, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 768), dtype=float32)</td>
			<td>shape : (1, 61, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 61, 64), dtype=float32)</td>
			<td>shape : (12, 61, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 61, 61), dtype=float32)</td>
			<td>shape : (1, 12, 61, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 61, 61), dtype=float32)</td>
			<td>shape : (12, 61, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 64, 61), dtype=float32)</td>
			<td>shape : (12, 64, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 61, 64), dtype=float32)</td>
			<td>shape : (1, 12, 61, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 12, 64), dtype=float32)</td>
			<td>shape : (61, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 61), dtype=float32)</td>
			<td>shape : (1, 12, 1, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 61), dtype=float32)</td>
			<td>shape : (12, 1, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 384), dtype=float32)</td>
			<td>shape : (1, 61, 6, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 61, 64), dtype=float32)</td>
			<td>shape : (6, 61, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 61, 61), dtype=float32)</td>
			<td>shape : (1, 6, 61, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 61, 61), dtype=float32)</td>
			<td>shape : (6, 61, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 64, 61), dtype=float32)</td>
			<td>shape : (6, 64, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 61, 64), dtype=float32)</td>
			<td>shape : (1, 6, 61, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 6, 64), dtype=float32)</td>
			<td>shape : (61, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 61), dtype=float32)</td>
			<td>shape : (1, 6, 1, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 61), dtype=float32)</td>
			<td>shape : (6, 1, 61)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 768), dtype=float32)</td>
			<td>shape : (1, 197, 12, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(729, 12), dtype=float32)</td>
			<td>shape : (1, 27, 27, 12)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 27, 27, 12), dtype=float32)</td>
			<td>shape : (729, 12)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(38809, 12), dtype=float32)</td>
			<td>shape : (197, 197, 12)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)</td>
			<td>shape : (1, 1024, 196, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1024), dtype=float32)</td>
			<td>shape : (197, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1024), dtype=float32)</td>
			<td>shape : (1, 197, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 1024), dtype=float32)</td>
			<td>shape : (1, 197, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 1024), dtype=float32)</td>
			<td>shape : (1, 197, 16, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 197, 64), dtype=float32)</td>
			<td>shape : (16, 197, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 197, 197), dtype=float32)</td>
			<td>shape : (1, 16, 197, 197)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(729, 16), dtype=float32)</td>
			<td>shape : (1, 27, 27, 16)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 27, 27, 16), dtype=float32)</td>
			<td>shape : (729, 16)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(38809, 16), dtype=float32)</td>
			<td>shape : (197, 197, 16)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 197, 197), dtype=float32)</td>
			<td>shape : (16, 197, 197)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 64, 197), dtype=float32)</td>
			<td>shape : (16, 64, 197)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 197, 64), dtype=float32)</td>
			<td>shape : (1, 16, 197, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 16, 64), dtype=float32)</td>
			<td>shape : (197, 1024)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2208, 1, 1), dtype=float32)</td>
			<td>shape : (1, 2208, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1664, 1, 1), dtype=float32)</td>
			<td>shape : (1, 1664, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1000, 1, 1), dtype=float32)</td>
			<td>shape : (1, 1000)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla34_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1000, 1, 1), dtype=float32)</td>
			<td>shape : (1, 1000, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(528, 1, 3, 3), dtype=float32)</td>
			<td>shape : (528, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(528, 1, 5, 5), dtype=float32)</td>
			<td>shape : (528, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(720, 1, 5, 5), dtype=float32)</td>
			<td>shape : (720, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1248, 1, 5, 5), dtype=float32)</td>
			<td>shape : (1248, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1248, 1, 3, 3), dtype=float32)</td>
			<td>shape : (1248, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(336, 1, 5, 5), dtype=float32)</td>
			<td>shape : (336, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1632, 1, 5, 5), dtype=float32)</td>
			<td>shape : (1632, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1632, 1, 3, 3), dtype=float32)</td>
			<td>shape : (1632, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(576, 1, 5, 5), dtype=float32)</td>
			<td>shape : (576, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(816, 1, 5, 5), dtype=float32)</td>
			<td>shape : (816, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1392, 1, 5, 5), dtype=float32)</td>
			<td>shape : (1392, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1392, 1, 3, 3), dtype=float32)</td>
			<td>shape : (1392, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(72, 1, 1, 5), dtype=float32)</td>
			<td>shape : (72, 1, 1, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(72, 1, 5, 1), dtype=float32)</td>
			<td>shape : (72, 1, 5, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(120, 1, 1, 5), dtype=float32)</td>
			<td>shape : (120, 1, 1, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(120, 1, 5, 1), dtype=float32)</td>
			<td>shape : (120, 1, 5, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(240, 1, 1, 5), dtype=float32)</td>
			<td>shape : (240, 1, 1, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(240, 1, 5, 1), dtype=float32)</td>
			<td>shape : (240, 1, 5, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(200, 1, 1, 5), dtype=float32)</td>
			<td>shape : (200, 1, 1, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(200, 1, 5, 1), dtype=float32)</td>
			<td>shape : (200, 1, 5, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(184, 1, 1, 5), dtype=float32)</td>
			<td>shape : (184, 1, 1, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(184, 1, 5, 1), dtype=float32)</td>
			<td>shape : (184, 1, 5, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(480, 1, 1, 5), dtype=float32)</td>
			<td>shape : (480, 1, 1, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(480, 1, 5, 1), dtype=float32)</td>
			<td>shape : (480, 1, 5, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(672, 1, 1, 5), dtype=float32)</td>
			<td>shape : (672, 1, 1, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(672, 1, 5, 1), dtype=float32)</td>
			<td>shape : (672, 1, 5, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(960, 1, 1, 5), dtype=float32)</td>
			<td>shape : (960, 1, 1, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(960, 1, 5, 1), dtype=float32)</td>
			<td>shape : (960, 1, 5, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 224, 224), dtype=float32)</td>
			<td>shape : (1, 224, 224)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 1, 1), dtype=float32)</td>
			<td>shape : (1, 2048)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 1, 1), dtype=float32)</td>
			<td>shape : (1, 1536)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 1, 1), dtype=float32)</td>
			<td>shape : (1, 1536, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 256, 256), dtype=float32)</td>
			<td>shape : (1, 3, 16, 16, 16, 16)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 16, 16, 16, 3), dtype=float32)</td>
			<td>shape : (256, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 512), dtype=float32)</td>
			<td>shape : (1, 256, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512), dtype=float32)</td>
			<td>shape : (1, 256, 512, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 256, 1), dtype=float32)</td>
			<td>shape : (1024, 256, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 512, 1), dtype=float32)</td>
			<td>shape : (1, 1024, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 512), dtype=float32)</td>
			<td>shape : (1, 1024, 512, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 1024, 1), dtype=float32)</td>
			<td>shape : (256, 1024, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512, 1), dtype=float32)</td>
			<td>shape : (1, 256, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mnist_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 12, 12), dtype=float32)</td>
			<td>shape : (1, 9216, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(72, 1, 3, 3), dtype=float32)</td>
			<td>shape : (72, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(120, 1, 5, 5), dtype=float32)</td>
			<td>shape : (120, 1, 5, 5)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(200, 1, 3, 3), dtype=float32)</td>
			<td>shape : (200, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(184, 1, 3, 3), dtype=float32)</td>
			<td>shape : (184, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)</td>
			<td>shape : (1, 960, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)</td>
			<td>shape : (1, 512)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 1, 1), dtype=float32)</td>
			<td>shape : (1, 440, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128, 128), dtype=float32)</td>
			<td>shape : (1, 32, 16384, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 32), dtype=float32)</td>
			<td>shape : (1, 16384, 1, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 32), dtype=float32)</td>
			<td>shape : (1, 128, 128, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16384), dtype=float32)</td>
			<td>shape : (1, 32, 128, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 16), dtype=float32)</td>
			<td>shape : (1, 32, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32), dtype=float32)</td>
			<td>shape : (256, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32), dtype=float32)</td>
			<td>shape : (1, 256, 1, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 32), dtype=float32)</td>
			<td>shape : (1, 256, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 256), dtype=float32)</td>
			<td>shape : (1, 1, 16384, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 16384, 256), dtype=float32)</td>
			<td>shape : (1, 16384, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 32, 256), dtype=float32)</td>
			<td>shape : (1, 32, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16384), dtype=float32)</td>
			<td>shape : (1, 128, 128, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128, 128), dtype=float32)</td>
			<td>shape : (1, 128, 16384, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 64, 64), dtype=float32)</td>
			<td>shape : (1, 64, 4096, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 64), dtype=float32)</td>
			<td>shape : (1, 4096, 2, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 64), dtype=float32)</td>
			<td>shape : (1, 64, 64, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 4096, 32), dtype=float32)</td>
			<td>shape : (2, 4096, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 4096), dtype=float32)</td>
			<td>shape : (1, 64, 64, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 16, 16), dtype=float32)</td>
			<td>shape : (1, 64, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64), dtype=float32)</td>
			<td>shape : (256, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64), dtype=float32)</td>
			<td>shape : (1, 256, 2, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 64), dtype=float32)</td>
			<td>shape : (1, 256, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 256, 32), dtype=float32)</td>
			<td>shape : (2, 256, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4096, 256), dtype=float32)</td>
			<td>shape : (1, 2, 4096, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 4096, 256), dtype=float32)</td>
			<td>shape : (2, 4096, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 32, 256), dtype=float32)</td>
			<td>shape : (2, 32, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4096, 32), dtype=float32)</td>
			<td>shape : (1, 2, 4096, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 2, 32), dtype=float32)</td>
			<td>shape : (4096, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4096, 64), dtype=float32)</td>
			<td>shape : (1, 4096, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)</td>
			<td>shape : (1, 256, 64, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64, 64), dtype=float32)</td>
			<td>shape : (1, 256, 4096, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 32, 32), dtype=float32)</td>
			<td>shape : (1, 160, 1024, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 160), dtype=float32)</td>
			<td>shape : (1, 1024, 5, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 160), dtype=float32)</td>
			<td>shape : (1, 32, 32, 160)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 1024, 32), dtype=float32)</td>
			<td>shape : (5, 1024, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 1024), dtype=float32)</td>
			<td>shape : (1, 160, 32, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 16, 16), dtype=float32)</td>
			<td>shape : (1, 160, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 160), dtype=float32)</td>
			<td>shape : (256, 160)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 160), dtype=float32)</td>
			<td>shape : (1, 256, 5, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 160), dtype=float32)</td>
			<td>shape : (1, 256, 160)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 256, 32), dtype=float32)</td>
			<td>shape : (5, 256, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 1024, 256), dtype=float32)</td>
			<td>shape : (1, 5, 1024, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 1024, 256), dtype=float32)</td>
			<td>shape : (5, 1024, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 32, 256), dtype=float32)</td>
			<td>shape : (5, 32, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 1024, 32), dtype=float32)</td>
			<td>shape : (1, 5, 1024, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 5, 32), dtype=float32)</td>
			<td>shape : (1024, 160)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 160), dtype=float32)</td>
			<td>shape : (1, 1024, 160)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 1024), dtype=float32)</td>
			<td>shape : (1, 640, 32, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(640, 1, 3, 3), dtype=float32)</td>
			<td>shape : (640, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 32, 32), dtype=float32)</td>
			<td>shape : (1, 640, 1024, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)</td>
			<td>shape : (1, 256, 256, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 32, 256), dtype=float32)</td>
			<td>shape : (8, 32, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 32), dtype=float32)</td>
			<td>shape : (1, 8, 256, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 32), dtype=float32)</td>
			<td>shape : (256, 256)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 256), dtype=float32)</td>
			<td>shape : (1, 1024, 16, 16)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 16, 16), dtype=float32)</td>
			<td>shape : (1, 1024, 256, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 38, 38), dtype=float32)</td>
			<td>shape : (1, 4, 5776)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 19, 19), dtype=float32)</td>
			<td>shape : (1, 4, 2166)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 10, 10), dtype=float32)</td>
			<td>shape : (1, 4, 600)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 5, 5), dtype=float32)</td>
			<td>shape : (1, 4, 150)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 3, 3), dtype=float32)</td>
			<td>shape : (1, 4, 36)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 324, 38, 38), dtype=float32)</td>
			<td>shape : (1, 81, 5776)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 486, 19, 19), dtype=float32)</td>
			<td>shape : (1, 81, 2166)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 486, 10, 10), dtype=float32)</td>
			<td>shape : (1, 81, 600)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 486, 5, 5), dtype=float32)</td>
			<td>shape : (1, 81, 150)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 324, 3, 3), dtype=float32)</td>
			<td>shape : (1, 81, 36)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 324, 1, 1), dtype=float32)</td>
			<td>shape : (1, 81, 4)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>shape : (3136, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>shape : (1, 3136, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 288), dtype=float32)</td>
			<td>shape : (64, 49, 288)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 288), dtype=float32)</td>
			<td>shape : (64, 49, 3, 3, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 3, 49, 32), dtype=float32)</td>
			<td>shape : (64, 3, 49, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 3, 49, 32), dtype=float32)</td>
			<td>shape : (192, 49, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 96), dtype=float32)</td>
			<td>shape : (1, 56, 56, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 96), dtype=float32)</td>
			<td>shape : (64, 49, 3, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 8, 7, 96), dtype=float32)</td>
			<td>shape : (1, 3136, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 384), dtype=float32)</td>
			<td>shape : (1, 56, 56, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 384), dtype=float32)</td>
			<td>shape : (3136, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 384), dtype=float32)</td>
			<td>shape : (784, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 384), dtype=float32)</td>
			<td>shape : (1, 784, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 192), dtype=float32)</td>
			<td>shape : (1, 28, 28, 192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>shape : (784, 192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>shape : (1, 784, 192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 576), dtype=float32)</td>
			<td>shape : (16, 49, 576)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 576), dtype=float32)</td>
			<td>shape : (16, 49, 3, 6, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 49, 32), dtype=float32)</td>
			<td>shape : (16, 6, 49, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 49, 32), dtype=float32)</td>
			<td>shape : (96, 49, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 192), dtype=float32)</td>
			<td>shape : (16, 49, 6, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 7, 4, 7, 192), dtype=float32)</td>
			<td>shape : (1, 784, 192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 768), dtype=float32)</td>
			<td>shape : (1, 28, 28, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 768), dtype=float32)</td>
			<td>shape : (784, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 768), dtype=float32)</td>
			<td>shape : (196, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 768), dtype=float32)</td>
			<td>shape : (1, 196, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 384), dtype=float32)</td>
			<td>shape : (1, 14, 14, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>shape : (196, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>shape : (1, 196, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 1152), dtype=float32)</td>
			<td>shape : (4, 49, 1152)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 1152), dtype=float32)</td>
			<td>shape : (4, 49, 3, 12, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 12, 49, 32), dtype=float32)</td>
			<td>shape : (4, 12, 49, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 12, 49, 32), dtype=float32)</td>
			<td>shape : (48, 49, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 384), dtype=float32)</td>
			<td>shape : (4, 49, 12, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 7, 2, 7, 384), dtype=float32)</td>
			<td>shape : (1, 196, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 1536), dtype=float32)</td>
			<td>shape : (1, 14, 14, 1536)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 1536), dtype=float32)</td>
			<td>shape : (196, 1536)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 1536), dtype=float32)</td>
			<td>shape : (49, 1536)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 1536), dtype=float32)</td>
			<td>shape : (1, 49, 1536)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 768), dtype=float32)</td>
			<td>shape : (1, 7, 7, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)</td>
			<td>shape : (1, 1, 7, 1, 7, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)</td>
			<td>shape : (49, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 7, 7, 768), dtype=float32)</td>
			<td>shape : (49, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 2304), dtype=float32)</td>
			<td>shape : (1, 49, 2304)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 2304), dtype=float32)</td>
			<td>shape : (1, 49, 3, 24, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 24, 49, 32), dtype=float32)</td>
			<td>shape : (1, 24, 49, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 24, 49, 32), dtype=float32)</td>
			<td>shape : (24, 49, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)</td>
			<td>shape : (1, 1, 1, 7, 7, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)</td>
			<td>shape : (49, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)</td>
			<td>shape : (1, 49, 24, 32)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)</td>
			<td>shape : (1, 49, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 1, 7, 768), dtype=float32)</td>
			<td>shape : (1, 7, 7, 768)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 3072), dtype=float32)</td>
			<td>shape : (1, 7, 7, 3072)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 3072), dtype=float32)</td>
			<td>shape : (49, 3072)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)</td>
			<td>shape : (1, 768, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
			<td>shape : (1, 96, 3136, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3136, 96), dtype=float32)</td>
			<td>shape : (1, 8, 7, 8, 7, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3136, 96), dtype=float32)</td>
			<td>shape : (1, 56, 56, 96)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=swin.encoder.layers.0.blocks.0.attention.self.relative_position_index, dtype=int64)</td>
			<td>shape : (2401,)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 192), dtype=float32)</td>
			<td>shape : (1, 4, 7, 4, 7, 192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 192), dtype=float32)</td>
			<td>shape : (1, 28, 28, 192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 384), dtype=float32)</td>
			<td>shape : (1, 2, 7, 2, 7, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 384), dtype=float32)</td>
			<td>shape : (1, 14, 14, 384)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1), dtype=float32)</td>
			<td>shape : (1, 768, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg11_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
			<td>shape : (1, 25088)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg11_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
			<td>shape : (1, 25088, 1, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(160, 1, 3, 3), dtype=float32)</td>
			<td>shape : (160, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(224, 1, 3, 3), dtype=float32)</td>
			<td>shape : (224, 1, 3, 3)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 85, 40, 40), dtype=float32)</td>
			<td>shape : (1, 255, 40, 40)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 255, 40, 40), dtype=float32)</td>
			<td>shape : (1, 1, 255, 1600)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 255, 1600), dtype=float32)</td>
			<td>shape : (1, 3, 85, 1600)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 1600, 85), dtype=float32)</td>
			<td>shape : (1, 4800, 85)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 85, 20, 20), dtype=float32)</td>
			<td>shape : (1, 255, 20, 20)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 255, 20, 20), dtype=float32)</td>
			<td>shape : (1, 1, 255, 400)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 255, 400), dtype=float32)</td>
			<td>shape : (1, 3, 85, 400)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 400, 85), dtype=float32)</td>
			<td>shape : (1, 1200, 85)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 85, 10, 10), dtype=float32)</td>
			<td>shape : (1, 255, 10, 10)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 255, 10, 10), dtype=float32)</td>
			<td>shape : (1, 1, 255, 100)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 255, 100), dtype=float32)</td>
			<td>shape : (1, 3, 85, 100)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 100, 85), dtype=float32)</td>
			<td>shape : (1, 300, 85)</td>
		</tr>
	</tbody>
</table>
