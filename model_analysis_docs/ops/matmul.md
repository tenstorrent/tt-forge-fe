<h1>Comprehensive Report on Matmul Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of matmul operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Matmul Operation Details</th>
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
			<td rowspan="919">1</td>
			<td rowspan="919">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="919">1882</td>
			<td>44</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>28</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 128, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 128, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 25088), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(25088, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4096, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4096, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4096, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 320), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 320), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 1024, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5, 64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 1024, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5, 256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 320), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 320), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 30000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 128, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 128, 4), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 4, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 196), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(196, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 196), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 128, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 128, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 16384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16384, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 128, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 256, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 14336), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14336), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14336, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 128256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 14336), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 14336), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14336, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 197, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 197), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 197, 197), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 197, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 13, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 64, 13), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 13, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 1, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 51200), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10240), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(10240, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 32, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 32, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 512, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 128, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 512, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1024, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 32, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9216), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(9216, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1001), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 32, 49), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 49, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 32, 49), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 49, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 32, 49), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 49, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 32, 49), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 49, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 64, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 1, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 64, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 384, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 384, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 384, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 30522), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 10, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 256, 10), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 10, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 131072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 7, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 7), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 7, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 50257), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 128, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 256, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 128256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 64, 4), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 4, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 32, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 64, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 32, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 32, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 80, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 256, 80), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 10240), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 51200), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(11, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 11, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 80, 11), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 11, 11), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 11, 80), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(11, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 10240), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 10240), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(10240, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 12, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 80, 12), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 12, 80), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 10240), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10240), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(10240, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 9216), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 96, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 256, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 32064), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 35, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 128, 35), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 35, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 8960), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 8960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8960, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 35, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(28, 128, 35), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(28, 35, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 18944), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 18944), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18944, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 152064), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 35, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 128, 35), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 35, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 11008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 61, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 61), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 61, 61), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 61, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 61), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 61), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 61, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 32128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 32128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 61, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 61), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 61, 61), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 61, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 61), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 61), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 61, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 32128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 197, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 197), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 197, 197), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 197, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1920), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1920, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1792, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 196), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(196, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 196), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4096, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 32, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4096, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 256, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4096, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 160), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 160), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 1024, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5, 32, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 1024, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5, 256, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 160), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 640), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(640, 160), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 256, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 288), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 576), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1152), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2304), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64, 13), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 13, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 64, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 64, 13), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 13, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 64, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 64, 13), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 13, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 6144), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 6144), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6144, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1500, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1500, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1500, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 51865), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 64, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1500, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 5120), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 5120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5120, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 5120), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 5120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5120, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 51865), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1500, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1500, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1500, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 51865), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1500, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1500, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1500, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 51865), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1500, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1500, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1500, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 51865), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 2, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 64, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 2, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 2, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 2, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 64, 1500), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 2, 1500), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1500, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 5120), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 5120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5120, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 51866), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 7, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 7), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 7, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(588, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 588, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 128, 588), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 588, 588), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 588, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(588, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 5504), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 588, 5504), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5504, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 588, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 32256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 39, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 128, 39), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 39, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 11008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 102400), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 577, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 577), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 577, 577), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 577, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 577, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_520, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 596, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 128, 596), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 596, 596), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 596, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 11008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 32064), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(204, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 204, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 204), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 204, 204), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 204, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 204, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 204, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(201, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 201, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 201), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 201, 201), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 201, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 3129), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 14, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 14), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 14, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(9, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 9, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 9), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 9, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 9, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 9), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 6, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 6), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 6, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 4608), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 96, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 32, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 6144), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 6144), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6144, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 250880), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 384, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 384, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 384, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 119547), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 9), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 28996), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 10, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 256, 10), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 10, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 131072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 4544), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4544, 18176), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 18176), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18176, 4544), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 4544), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4544, 4672), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(71, 6, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 6), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(71, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 6, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 4544), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4544, 4544), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 4544), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4544, 65024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 9216), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 9216), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(9216, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 23040), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 23040), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(23040, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 12288), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 334, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64, 334), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 334, 334), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 334, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(334, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 16384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 16384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16384, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 207, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 256, 207), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 207, 207), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 207, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 14336), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 14336), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14336, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 256000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 7, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 256, 7), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 7, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 16384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 16384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16384, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 256000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 2304), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2304, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 2304), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2304, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 207, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 256, 207), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 207, 207), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 207, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2304), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(207, 2304), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2304, 9216), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 9216), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(9216, 2304), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 2304), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2304, 256000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(107, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(107, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 107, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 256, 107), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 107, 107), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 107, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(107, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 16384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 16384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16384, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 256000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(107, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 107, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 256, 107), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 107, 107), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 107, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(107, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(107, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 24576), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 24576), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24576, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 256000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 128, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 32, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 256, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 128, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 256, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 10240), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 50257), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 32, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 128, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 32, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 10240), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 10240), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(10240, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2560, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 50257), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 11008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 4, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 128, 4), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 4, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 32, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 128, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 32, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 128256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 32, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 128, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 32, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 11008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 32000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 14336), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14336), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14336, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 32000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 50272), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 50272), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 50272), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(50176, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 50176), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 50176), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 50176, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 322), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3025, 322), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(322, 322), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 322), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 322, 3025), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 3025), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3025, 322), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 322), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(322, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 261), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(50176, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(261, 261), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 261, 50176), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 50176), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(261, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 32, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 2048, 160), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 256, 160), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 2048, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 32, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 2048, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 256, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 262), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 9216), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 5, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 96, 5), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 5, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 9216), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 13, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 96, 13), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 13, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(13, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(13, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 8192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 29), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 29, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 2816), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 2816), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2816, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 64, 6), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 6, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 2816), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 2816), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2816, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 896), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 35, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14, 64, 35), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14, 35, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(35, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 4864), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 4864), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4864, 896), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 39, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 128, 39), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 39, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 11008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 29, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 128, 29), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 29, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 11008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 29, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 128, 29), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 29, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 8960), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 8960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8960, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 39, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 128, 39), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 39, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 8960), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 8960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8960, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 896), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 39, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14, 64, 39), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14, 39, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 4864), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 4864), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4864, 896), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 39, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(28, 128, 39), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(28, 39, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(39, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 18944), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 18944), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18944, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 152064), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 896), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14, 64, 29), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14, 29, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 4864), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 4864), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4864, 896), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 896), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(896, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(13, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(13, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 13, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(28, 128, 13), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(28, 13, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(13, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 18944), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 18944), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18944, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 29, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(28, 128, 29), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(28, 29, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(29, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 18944), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 18944), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18944, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 3584), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3584, 152064), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 250002), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 61, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 64, 61), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 61, 61), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 61, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 64, 61), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 61), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 61, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 2816), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 2816), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2816, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 2816), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 2816), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2816, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 61, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 64, 61), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 61, 61), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 61, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(61, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 64, 61), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 61), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 61, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 256008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 256008), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_generic_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 72), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_generic_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_generic_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 72), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 8), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.forecast_time, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.backcast_time, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 72), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 48), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.forecast_sin_template, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.forecast_cos_template, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.backcast_sin_template, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.backcast_cos_template, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(784, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 12), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 3), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3, 12), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_linear_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 784), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 197, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 64, 197), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 197, 197), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(6, 197, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(197, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 197, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3, 64, 197), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 197, 197), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3, 197, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2208), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2208, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 18), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1664), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1664, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(300, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 300), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 300), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 300, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(300, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4800, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 64, 300), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4800, 300), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 300, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4800, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 320), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(300, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 320), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 1200, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5, 64, 300), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(5, 1200, 300), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(5, 300, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1200, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 320), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 320), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(300, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 300, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 64, 300), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 300, 300), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 300, 64), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 11221), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 21843), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mnist_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9216), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(9216, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mnist_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 10), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 9), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1001), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1001), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1280), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1088, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7392), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(7392, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(888, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3712), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3712, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(440, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2520), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2520, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1008, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 912), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(912, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2016, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(784, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1512, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 400), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(400, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3024, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(320, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1000), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 32, 49), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 49, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3136, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 32, 49), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 49, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 32, 49), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 49, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 49, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 32, 49), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 49, 49), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 49, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(49, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3136, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 96), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 192), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 768), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 3072), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 768), dtype=float32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
