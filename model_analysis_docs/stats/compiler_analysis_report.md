<h1>Compiler Component Failure Analysis by Model Impacts</h1>
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
			<td rowspan="8">Metalium</td>
			<td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td>128</td>
			<td><ul><li>pt_hrnet_hrnet_w18_timm</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls</li><li>pt_phi2_microsoft_phi_2_seq_cls</li><li>pt_vovnet_ese_vovnet99b</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm</li><li>pt_albert_xxlarge_v2_token_cls</li><li>pt_whisper_openai_whisper_tiny</li><li>pt_segformer_nvidia_mit_b2_img_cls</li><li>pt_vgg_vgg16</li><li>pt_gemma_google_gemma_2b</li><li>pt_whisper_openai_whisper_small</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm</li><li>pt_phi2_microsoft_phi_2_pytdml_clm</li><li>pt_stereo_facebook_musicgen_small</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg</li><li>pt_alexnet_base_osmr</li><li>pt_dpr_facebook_dpr_reader_multiset_base_reader</li><li>pt_hrnet_hrnet_w64_timm</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_reader</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg</li><li>pt_rcnn_base_rect_0</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li><li>pt_unet_qubvel</li><li>pt_hrnet_hrnet_w18_small_v2_timm</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls</li><li>pt_vgg_vgg19</li><li>pt_albert_xxlarge_v1_token_cls</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm</li><li>pt_hrnet_hrnetv2_w30_osmr</li><li>pt_hrnet_hrnet_w18_small_timm</li><li>pt_opt_facebook_opt_1_3b_seq_cls</li><li>pt_opt_facebook_opt_125m_seq_cls</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls</li><li>pt_mobilenetv2_basic_torchhub</li><li>pt_mistral_mistralai_mistral_7b_v0_1</li><li>pt_vgg_vgg13</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm</li><li>pt_hrnet_hrnetv2_w48_osmr</li><li>pt_t5_t5_large_text_gen</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls</li><li>pt_whisper_openai_whisper_medium</li><li>pt_stereo_facebook_musicgen_large</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm</li><li>pt_whisper_openai_whisper_large</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm</li><li>pt_vgg_bn_vgg19</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls</li><li>pt_t5_google_flan_t5_base_text_gen</li><li>pt_efficientnet_efficientnet_b0_timm</li><li>pt_vgg_vgg11</li><li>pt_segformer_nvidia_mit_b0_img_cls</li><li>pt_t5_google_flan_t5_small_text_gen</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls</li><li>pt_dla_dla169</li><li>pt_xception_xception65_timm</li><li>pt_albert_xxlarge_v1_mlm</li><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls</li><li>pt_albert_xxlarge_v2_mlm</li><li>pt_mobilnetv3_mobilenetv3_small_100_timm</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm</li><li>pt_hrnet_hrnet_w30_timm</li><li>pt_vovnet_ese_vovnet39b</li><li>pt_alexnet_alexnet_torchhub</li><li>pt_fuyu_adept_fuyu_8b</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls</li><li>pt_hrnet_hrnetv2_w32_osmr</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm</li><li>pt_xception_xception71_timm</li><li>pt_whisper_openai_whisper_base</li><li>pt_hrnet_hrnet_w48_timm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls</li><li>pt_opt_facebook_opt_350m_seq_cls</li><li>pt_segformer_nvidia_mit_b5_img_cls</li><li>pt_vgg_bn_vgg19b</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls</li><li>pt_xception_xception41_timm</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li><li>pt_hrnet_hrnet_w32_timm</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm</li><li>pt_t5_google_flan_t5_large_text_gen</li><li>pt_segformer_nvidia_mit_b1_img_cls</li><li>pt_segformer_nvidia_mit_b3_img_cls</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls</li><li>pt_phi2_microsoft_phi_2_clm</li><li>pt_hrnet_hrnetv2_w18_osmr</li><li>pt_falcon_tiiuae_falcon_7b_instruct</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm</li><li>pt_vovnet_ese_vovnet19b_dw</li><li>pt_t5_t5_small_text_gen</li><li>pt_phi2_microsoft_phi_2_token_cls</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg</li><li>pt_stereo_facebook_musicgen_medium</li><li>pt_vgg_vgg19_bn_timm</li><li>pt_hrnet_hrnet_w44_timm</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm</li><li>pt_hrnet_hrnet_w18_small_v2_osmr</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls</li><li>pt_hrnet_hrnet_w18_small_v1_osmr</li><li>pt_ghostnet_ghostnet_100_timm</li><li>pt_hrnet_hrnetv2_w64_osmr</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_segformer_nvidia_mit_b4_img_cls</li><li>pt_vgg_vgg19_bn_torchhub</li><li>pt_hrnet_hrnetv2_w44_osmr</li><li>pt_vgg_19_hf</li><li>pt_hrnet_hrnet_w40_timm</li><li>pt_t5_t5_base_text_gen</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b4_timm</li><li>pt_retinanet_retinanet_rn152fpn</li><li>pt_mobilnetv3_mobilenetv3_large_100_timm</li><li>pt_hrnet_hrnetv2_w40_osmr</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn.reshape] RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
			<td>77</td>
			<td><ul><li>pt_hrnet_hrnet_w18_timm</li><li>pt_vovnet_ese_vovnet99b</li><li>pt_resnext_resnext26_32x4d_osmr</li><li>pt_vovnet_vovnet27s_osmr</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li><li>pt_densenet_densenet161</li><li>pt_dla_dla102x2</li><li>pt_alexnet_base_osmr</li><li>pt_densenet_densenet169</li><li>pt_xception_xception_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_hf</li><li>pt_hrnet_hrnet_w64_timm</li><li>pt_resnext_resnext14_32x4d_osmr</li><li>pt_vovnet_vovnet39_osmr</li><li>pt_wideresnet_wide_resnet101_2_timm</li><li>pt_mobilenet_v1_basic</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li><li>pt_hrnet_hrnet_w18_small_v2_timm</li><li>pt_hrnet_hrnetv2_w30_osmr</li><li>pt_hrnet_hrnet_w18_small_timm</li><li>pt_mobilenetv2_basic_torchhub</li><li>pt_resnext_resnext50_32x4d_torchhub</li><li>pt_hrnet_hrnetv2_w48_osmr</li><li>pt_dla_dla102</li><li>pt_densenet_densenet121</li><li>pt_efficientnet_efficientnet_b0_timm</li><li>pt_dla_dla102x</li><li>pt_dla_dla169</li><li>pt_xception_xception65_timm</li><li>pt_wideresnet_wide_resnet50_2_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_timm</li><li>pt_hrnet_hrnet_w30_timm</li><li>pt_vovnet_ese_vovnet39b</li><li>pt_resnet_50_timm</li><li>pt_dla_dla46_c</li><li>pt_hrnet_hrnetv2_w32_osmr</li><li>pt_xception_xception71_timm</li><li>pt_inception_v4_osmr</li><li>pt_dla_dla34</li><li>pt_hrnet_hrnet_w48_timm</li><li>pt_resnext_resnext50_32x4d_osmr</li><li>pt_googlenet_base</li><li>pt_inception_v4_timm</li><li>pt_mobilenetv2_mobilenetv2_100_timm</li><li>pt_dla_dla46x_c</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li><li>pt_efficientnet_efficientnet_b4_torchvision</li><li>pt_xception_xception41_timm</li><li>pt_hrnet_hrnet_w32_timm</li><li>pt_densenet_densenet201</li><li>pt_regnet_facebook_regnet_y_040_img_cls</li><li>pt_dla_dla60x</li><li>pt_hrnet_hrnetv2_w18_osmr</li><li>pt_wideresnet_wide_resnet50_2</li><li>pt_dla_dla60x_c</li><li>pt_vovnet_ese_vovnet19b_dw</li><li>pt_resnext_resnext101_64x4d_osmr</li><li>pt_vovnet_vovnet57_osmr</li><li>pt_dla_dla60</li><li>pt_hrnet_hrnet_w44_timm</li><li>pt_resnext_resnext101_32x8d_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_osmr</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_hf</li><li>pt_hrnet_hrnet_w18_small_v1_osmr</li><li>pt_regnet_facebook_regnet_y_040</li><li>pt_ghostnet_ghostnet_100_timm</li><li>pt_hrnet_hrnetv2_w64_osmr</li><li>pt_efficientnet_efficientnet_b0_torchvision</li><li>pt_hrnet_hrnetv2_w44_osmr</li><li>pt_wideresnet_wide_resnet101_2</li><li>pt_hrnet_hrnet_w40_timm</li><li>pt_resnet_50_hf</li><li>pt_efficientnet_efficientnet_b4_timm</li><li>pt_resnext_resnext101_32x8d_wsl_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_timm</li><li>pt_hrnet_hrnetv2_w40_osmr</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
			<td>57</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li><li>pt_rcnn_base_rect_0</li><li>pt_yolo_v6_yolov6s</li><li>pt_unet_qubvel</li><li>pt_alexnet_alexnet_torchhub</li><li>pt_segformer_nvidia_mit_b3_img_cls</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li><li>pt_vgg_vgg19</li><li>pt_fuyu_adept_fuyu_8b</li><li>pt_vovnet_ese_vovnet99b</li><li>pt_yolo_v6_yolov6l</li><li>pt_vgg_vgg13</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li><li>pt_yolox_yolox_l</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg</li><li>pt_yolo_v6_yolov6m</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg</li><li>pt_unet_cityscape_osmr</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg</li><li>pt_segformer_nvidia_mit_b2_img_cls</li><li>pt_stereo_facebook_musicgen_medium</li><li>pt_vgg_vgg16</li><li>pt_ssd300_resnet50_base</li><li>pt_vgg_vgg19_bn_timm</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls</li><li>pt_unet_base</li><li>pt_stereo_facebook_musicgen_large</li><li>pt_yolox_yolox_x</li><li>pt_yolox_yolox_m</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li><li>pt_vgg_bn_vgg19</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li><li>pt_efficientnet_efficientnet_b0_timm</li><li>pt_segformer_nvidia_mit_b5_img_cls</li><li>pt_vgg_bn_vgg19b</li><li>pt_vgg_vgg11</li><li>pt_segformer_nvidia_mit_b0_img_cls</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li><li>pt_efficientnet_efficientnet_b0_torchvision</li><li>pt_alexnet_base_osmr</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg</li><li>pt_efficientnet_efficientnet_b4_torchvision</li><li>pt_monodle_base</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_segformer_nvidia_mit_b4_img_cls</li><li>pt_vgg_vgg19_bn_torchhub</li><li>pt_vgg_19_hf</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_yolox_yolox_darknet</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li><li>pt_segformer_nvidia_mit_b1_img_cls</li><li>pt_efficientnet_efficientnet_b4_timm</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg</li><li>pt_yolo_v6_yolov6n</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
			<td>30</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_opt_facebook_opt_125m_qa</li><li>pt_distilbert_distilbert_base_uncased_mlm</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_opt_facebook_opt_1_3b_seq_cls</li><li>pt_opt_facebook_opt_125m_seq_cls</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_opt_facebook_opt_350m_qa</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm</li><li>pt_opt_facebook_opt_1_3b_qa</li><li>pt_stereo_facebook_musicgen_medium</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa</li><li>pt_opt_facebook_opt_350m_clm</li><li>pt_opt_facebook_opt_350m_seq_cls</li><li>pt_stereo_facebook_musicgen_large</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm</li><li>pt_stereo_facebook_musicgen_small</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm</li><li>pt_xglm_facebook_xglm_1_7b_clm</li><li>pt_distilbert_distilbert_base_cased_mlm</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm</li><li>pt_clip_openai_clip_vit_base_patch32_text</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttmetal allocations] RuntimeError tt-metal/tt_metal/impl/allocator/allocator.cpp Out of Memory: Not enough space to allocate</td>
			<td>21</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li><li>pt_retinanet_retinanet_rn18fpn</li><li>pt_retinanet_retinanet_rn50fpn</li><li>pt_inception_v4_osmr</li><li>pt_inception_v4_timm</li><li>pt_yolox_yolox_x</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls</li><li>pt_deit_facebook_deit_small_patch16_224_img_cls</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li><li>pt_retinanet_retinanet_rn101fpn</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls</li><li>pt_retinanet_retinanet_rn34fpn</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_yolox_yolox_darknet</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_retinanet_retinanet_rn152fpn</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast</td>
			<td>11</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_opt_facebook_opt_350m_seq_cls</li><li>pt_opt_facebook_opt_125m_qa</li><li>pt_opt_facebook_opt_1_3b_qa</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_xglm_facebook_xglm_1_7b_clm</li><li>pt_opt_facebook_opt_350m_clm</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_opt_facebook_opt_1_3b_seq_cls</li><li>pt_opt_facebook_opt_125m_seq_cls</li><li>pt_opt_facebook_opt_350m_qa</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn softmax] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B Inputs must be of bfloat16 or bfloat8_b type</td>
			<td>2</td>
			<td><ul><li>pt_yolo_v6_yolov6m</li><li>pt_yolo_v6_yolov6l</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][tt-metal ncrisc build] RuntimeError tt-metal/tt_metal/impl/program/program.cpp Failed to generate binaries for reader_conv_activations_padded_with_halo_3x3_weights_v2 ncrisc build failed</td>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base</li></ul></td>
		</tr>
		<tr>
			<td rowspan="70">Forge-Fe</td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>97</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls</li><li>pt_phi2_microsoft_phi_2_seq_cls</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm</li><li>pt_albert_xxlarge_v2_token_cls</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder</li><li>pt_gemma_google_gemma_2b</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm</li><li>pt_stereo_facebook_musicgen_small</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_question_encoder</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm</li><li>pt_albert_xlarge_v1_mlm</li><li>pt_dpr_facebook_dpr_reader_multiset_base_reader</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_reader</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_question_encoder</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls</li><li>pt_albert_xxlarge_v1_token_cls</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm</li><li>pt_albert_base_v2_token_cls</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm</li><li>pt_opt_facebook_opt_1_3b_seq_cls</li><li>pt_opt_facebook_opt_125m_seq_cls</li><li>pt_albert_large_v2_mlm</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls</li><li>pt_mistral_mistralai_mistral_7b_v0_1</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm</li><li>pt_t5_t5_large_text_gen</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa</li><li>pt_stereo_facebook_musicgen_large</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls</li><li>pt_distilbert_distilbert_base_cased_mlm</li><li>pt_t5_google_flan_t5_base_text_gen</li><li>pt_t5_google_flan_t5_small_text_gen</li><li>pt_albert_xlarge_v2_mlm</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls</li><li>pt_clip_openai_clip_vit_base_patch32_text</li><li>pt_albert_large_v2_token_cls</li><li>pt_albert_xxlarge_v1_mlm</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls</li><li>pt_albert_xlarge_v2_token_cls</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls</li><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_albert_xxlarge_v2_mlm</li><li>pt_opt_facebook_opt_125m_qa</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm</li><li>pt_distilbert_distilbert_base_uncased_mlm</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li><li>pt_albert_base_v2_mlm</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_opt_facebook_opt_350m_clm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls</li><li>pt_opt_facebook_opt_350m_seq_cls</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li><li>pt_albert_large_v1_token_cls</li><li>pt_albert_xlarge_v1_token_cls</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_t5_google_flan_t5_large_text_gen</li><li>pt_albert_base_v1_token_cls</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls</li><li>pt_opt_facebook_opt_350m_qa</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</li><li>pt_falcon_tiiuae_falcon_7b_instruct</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm</li><li>pt_opt_facebook_opt_1_3b_qa</li><li>pt_bert_bert_base_uncased_mlm</li><li>pt_t5_t5_small_text_gen</li><li>pt_phi2_microsoft_phi_2_token_cls</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls</li><li>pt_stereo_facebook_musicgen_medium</li><li>pt_albert_base_v1_mlm</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm</li><li>pt_xglm_facebook_xglm_1_7b_clm</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_albert_large_v1_mlm</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li><li>pt_t5_t5_base_text_gen</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>91</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat</li><li>pt_phi2_microsoft_phi_2_seq_cls</li><li>pt_albert_xxlarge_v2_token_cls</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm</li><li>pt_gemma_google_gemma_2b</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm</li><li>pt_phi2_microsoft_phi_2_pytdml_clm</li><li>pt_stereo_facebook_musicgen_small</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm</li><li>pt_albert_xlarge_v1_mlm</li><li>pt_dpr_facebook_dpr_reader_multiset_base_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_reader</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls</li><li>pt_albert_xxlarge_v1_token_cls</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm</li><li>pt_albert_base_v2_token_cls</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm</li><li>pt_opt_facebook_opt_1_3b_seq_cls</li><li>pt_opt_facebook_opt_125m_seq_cls</li><li>pt_albert_large_v2_mlm</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls</li><li>pt_mistral_mistralai_mistral_7b_v0_1</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls</li><li>pt_stereo_facebook_musicgen_large</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls</li><li>pt_albert_xlarge_v2_mlm</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls</li><li>pt_clip_openai_clip_vit_base_patch32_text</li><li>pt_albert_large_v2_token_cls</li><li>pt_albert_xxlarge_v1_mlm</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls</li><li>pt_albert_xlarge_v2_token_cls</li><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_albert_xxlarge_v2_mlm</li><li>pt_opt_facebook_opt_125m_qa</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm</li><li>pt_fuyu_adept_fuyu_8b</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_albert_base_v2_mlm</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls</li><li>pt_opt_facebook_opt_350m_clm</li><li>pt_opt_facebook_opt_350m_seq_cls</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls</li><li>pt_albert_large_v1_token_cls</li><li>pt_albert_xlarge_v1_token_cls</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls</li><li>pt_albert_base_v1_token_cls</li><li>pt_phi2_microsoft_phi_2_clm</li><li>pt_opt_facebook_opt_350m_qa</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</li><li>pt_falcon_tiiuae_falcon_7b_instruct</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm</li><li>pt_opt_facebook_opt_1_3b_qa</li><li>pt_bert_bert_base_uncased_mlm</li><li>pt_phi2_microsoft_phi_2_token_cls</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls</li><li>pt_stereo_facebook_musicgen_medium</li><li>pt_albert_base_v1_mlm</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm</li><li>pt_xglm_facebook_xglm_1_7b_clm</li><li>pt_deit_facebook_deit_small_patch16_224_img_cls</li><li>pt_albert_large_v1_mlm</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: resize2d</td>
			<td>55</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li><li>pt_hrnet_hrnet_w18_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li><li>pt_unet_qubvel</li><li>pt_hrnet_hrnet_w30_timm</li><li>pt_hrnet_hrnet_w18_small_v2_timm</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li><li>pt_retinanet_retinanet_rn18fpn</li><li>pt_hrnet_hrnetv2_w30_osmr</li><li>pt_hrnet_hrnet_w18_small_timm</li><li>pt_yolox_yolox_tiny</li><li>pt_hrnet_hrnetv2_w18_osmr</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_320x320</li><li>pt_retinanet_retinanet_rn50fpn</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li><li>pt_yolox_yolox_l</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg</li><li>pt_hrnet_hrnetv2_w48_osmr</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg</li><li>pt_hrnet_hrnetv2_w32_osmr</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg</li><li>pt_unet_cityscape_osmr</li><li>pt_retinanet_retinanet_rn152fpn</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li><li>pt_hrnet_hrnet_w44_timm</li><li>pt_hrnet_hrnet_w48_timm</li><li>pt_yolox_yolox_x</li><li>pt_hrnet_hrnet_w18_small_v2_osmr</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li><li>pt_yolox_yolox_m</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li><li>pt_yolox_yolox_nano</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li><li>pt_hrnet_hrnet_w18_small_v1_osmr</li><li>pt_retinanet_retinanet_rn101fpn</li><li>pt_hrnet_hrnetv2_w64_osmr</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg</li><li>pt_retinanet_retinanet_rn34fpn</li><li>pt_fpn_base_torchvision</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_hrnet_hrnetv2_w44_osmr</li><li>pt_hrnet_hrnet_w32_timm</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_yolox_yolox_darknet</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li><li>pt_hrnet_hrnet_w40_timm</li><li>pt_hrnet_hrnet_w64_timm</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_320x320</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg</li><li>pt_yolox_yolox_s</li><li>pt_hrnet_hrnetv2_w40_osmr</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td>36</td>
			<td><ul><li>pt_albert_xxlarge_v2_mlm</li><li>pt_albert_base_v1_token_cls</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_question_encoder</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls</li><li>pt_albert_xxlarge_v1_token_cls</li><li>pt_distilbert_distilbert_base_uncased_mlm</li><li>pt_albert_base_v2_token_cls</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_albert_large_v2_mlm</li><li>pt_albert_base_v2_mlm</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls</li><li>pt_albert_xxlarge_v2_token_cls</li><li>pt_bert_bert_base_uncased_mlm</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder</li><li>pt_albert_base_v1_mlm</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa</li><li>pt_distilbert_distilbert_base_cased_mlm</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_question_encoder</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls</li><li>pt_albert_xlarge_v1_mlm</li><li>pt_albert_xlarge_v2_mlm</li><li>pt_dpr_facebook_dpr_reader_multiset_base_reader</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls</li><li>pt_albert_large_v1_mlm</li><li>pt_albert_large_v1_token_cls</li><li>pt_albert_xlarge_v1_token_cls</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li><li>pt_clip_openai_clip_vit_base_patch32_text</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_reader</li><li>pt_albert_large_v2_token_cls</li><li>pt_albert_xxlarge_v1_mlm</li><li>pt_albert_xlarge_v2_token_cls</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td>17</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm</li><li>pt_opt_facebook_opt_350m_seq_cls</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_dpr_facebook_dpr_reader_multiset_base_reader</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls</li><li>pt_opt_facebook_opt_125m_qa</li><li>pt_opt_facebook_opt_1_3b_qa</li><li>pt_opt_facebook_opt_350m_qa</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_reader</li><li>pt_distilbert_distilbert_base_cased_mlm</li><li>pt_distilbert_distilbert_base_uncased_mlm</li><li>pt_opt_facebook_opt_1_3b_seq_cls</li><li>pt_opt_facebook_opt_125m_seq_cls</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls</li><li>pt_alexnet_base_osmr</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: cumsum</td>
			<td>11</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm</li><li>pt_opt_facebook_opt_350m_seq_cls</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls</li><li>pt_opt_facebook_opt_125m_qa</li><li>pt_opt_facebook_opt_1_3b_qa</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_opt_facebook_opt_350m_qa</li><li>pt_opt_facebook_opt_1_3b_seq_cls</li><li>pt_opt_facebook_opt_125m_seq_cls</li><li>pt_opt_facebook_opt_350m_clm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: conv2d_transpose</td>
			<td>7</td>
			<td><ul><li>pt_yolo_v6_yolov6l</li><li>pt_unet_base</li><li>pt_monodle_base</li><li>pt_yolo_v6_yolov6s</li><li>pt_yolo_v6_yolov6m</li><li>pt_autoencoder_conv</li><li>pt_yolo_v6_yolov6n</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype Unsupported] RuntimeError Unhandled dtype Bool</td>
			<td>7</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1152, 384]</td>
			<td>5</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li><li>pt_mobilenetv2_mobilenetv2_100_timm</li><li>pt_mobilenetv2_basic_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [1225, 1], got [0, 0]</td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_osmr</li><li>pt_inception_v4_timm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [289, 1], got [0, 0]</td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_osmr</li><li>pt_inception_v4_timm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [64, 1], got [0, 0]</td>
			<td>2</td>
			<td><ul><li>pt_inception_v4_osmr</li><li>pt_inception_v4_timm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [16, 1, 1, 1], got [1, 96, 1536, 1536]</td>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 24, 2304, 2304]</td>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 144, 3456, 3456]</td>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 24, 3456, 3456]</td>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 48, 1152, 1152]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 96, 4608, 4608]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 96, 9216, 9216]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 192, 18432, 18432]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 192, 36864, 36864]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 384, 73728, 73728]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 384, 147456, 147456]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 768, 294912, 294912]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [768, 1, 1, 1], got [1, 768, 589824, 589824]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [16, 1, 1, 1], got [1, 8, 128, 128]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [8, 1, 1, 1], got [1, 48, 384, 384]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 8, 384, 384]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 16, 768, 768]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 16, 1536, 1536]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 32, 4608, 4608]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [32, 1, 1, 1], got [1, 192, 6144, 6144]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 32, 6144, 6144]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 56, 10752, 10752]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [56, 1, 1, 1], got [1, 336, 18816, 18816]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [336, 1, 1, 1], got [1, 56, 18816, 18816]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [336, 1, 1, 1], got [1, 112, 37632, 37632]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [112, 1, 1, 1], got [1, 1280, 143360, 143360]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 64, 12288, 12288]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [64, 1, 1, 1], got [1, 384, 24576, 24576]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 64, 24576, 24576]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 96, 36864, 36864]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 576, 55296, 55296]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [576, 1, 1, 1], got [1, 96, 55296, 55296]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [576, 1, 1, 1], got [1, 160, 92160, 92160]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [160, 1, 1, 1], got [1, 960, 153600, 153600]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [960, 1, 1, 1], got [1, 160, 153600, 153600]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [960, 1, 1, 1], got [1, 320, 307200, 307200]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [320, 1, 1, 1], got [1, 256, 81920, 81920]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [256, 1, 1, 1], got [1, 21, 5376, 5376]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 16, 384, 384]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 48, 6912, 6912]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 288, 13824, 13824]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [288, 1, 1, 1], got [1, 48, 13824, 13824]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [288, 1, 1, 1], got [1, 72, 20736, 20736]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [72, 1, 1, 1], got [1, 432, 31104, 31104]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [432, 1, 1, 1], got [1, 72, 31104, 31104]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [432, 1, 1, 1], got [1, 120, 51840, 51840]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [120, 1, 1, 1], got [1, 720, 86400, 86400]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [720, 1, 1, 1], got [1, 120, 86400, 86400]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [720, 1, 1, 1], got [1, 240, 172800, 172800]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [240, 1, 1, 1], got [1, 1280, 307200, 307200]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1], got [1, 12]</td>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [72, 1], got [1, 12]</td>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2304, 768]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 864, 288]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1296, 432]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2160, 720]</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [768, 1], got [1, 1001]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [1024, 1], got [1, 1001]</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_hf</li></ul></td>
		</tr>
		<tr>
			<td rowspan="2">MLIR</td>
			<td>[MLIR][MLIR runtime ttnn ] tt::exception tt-mlir/runtime/lib/ttnn/runtime.cpp Unsupported data type</td>
			<td>19</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls</li><li>pt_distilbert_distilbert_base_uncased_mlm</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm</li><li>pt_distilbert_distilbert_base_cased_mlm</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls</li></ul></td>
		</tr>
		<tr>
			<td>[MLIR][TTIR to TTNN Conv2dOpConversionPattern] tt_forge_signal_handler tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &) adaptor.getPaddingBottom() == adaptor.getPaddingTop() TTNN only supports padding height/width attributes. Thus, padding_top must equal padding_bottom for the op to execute as expected</td>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</li></ul></td>
		</tr>
	<tbody>
</table>
<h1>Compiler-Specific Model Statistics</h1>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Total no of models : 268</th>
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
			<td>86</td>
			<td>32 %</td>
			<td>96 %</td>
			<td><ul><li>pt_autoencoder_conv (84 %)</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa (84 %)</li><li>pt_nbeats_seasionality_basis (86 %)</li><li>pt_bert_bert_base_uncased_mlm (87 %)</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls (87 %)</li><li>pt_dpr_facebook_dpr_reader_multiset_base_reader (87 %)</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_reader (87 %)</li><li>pt_albert_xlarge_v1_token_cls (88 %)</li><li>pt_albert_xlarge_v2_token_cls (88 %)</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls (88 %)</li></ul></td>
		</tr>
		<tr>
			<td>MLIR</td>
			<td>85</td>
			<td>32 %</td>
			<td>96 %</td>
			<td><ul><li>pt_autoencoder_conv (84 %)</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa (84 %)</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa (85 %)</li><li>pt_nbeats_seasionality_basis (86 %)</li><li>pt_bert_bert_base_uncased_mlm (87 %)</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls (87 %)</li><li>pt_dpr_facebook_dpr_reader_multiset_base_reader (87 %)</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_reader (87 %)</li><li>pt_albert_xlarge_v1_token_cls (88 %)</li><li>pt_albert_xlarge_v2_token_cls (88 %)</li></ul></td>
		</tr>
		<tr>
			<td>Metalium</td>
			<td>31</td>
			<td>12 %</td>
			<td>94 %</td>
			<td><ul><li>pt_alexnet_base_osmr (83 %)</li><li>pt_opt_facebook_opt_125m_seq_cls (83 %)</li><li>pt_opt_facebook_opt_1_3b_seq_cls (83 %)</li><li>pt_autoencoder_conv (84 %)</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa (84 %)</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa (84 %)</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm (84 %)</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm (84 %)</li><li>pt_opt_facebook_opt_350m_seq_cls (84 %)</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls (84 %)</li></ul></td>
		</tr>
	<tbody>
</table>
