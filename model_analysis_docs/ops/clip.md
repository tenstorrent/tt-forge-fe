<h1>Comprehensive Report on Clip Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of clip operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Clip Operation Details</th>
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
			<td rowspan="142">1</td>
			<td rowspan="142">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="142">291</td>
			<td>13</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td>min : 0.0<br>max : 1.0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=float32)</td>
			<td>min : 0.0<br>max : 1.0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
			<td>min : 0.0<br>max : 1.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1, 13), dtype=float32)</td>
			<td>min : 0.0<br>max : 1.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 200, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 184, 14, 14), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 1.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 384, 384), dtype=float32)</td>
			<td>min : 0.0<br>max : 1.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 120, 120), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 120, 120), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 60, 60), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 60, 60), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 30, 30), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 30, 30), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 15, 15), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 15, 15), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 15, 15), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 8, 8), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 8, 8), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 8, 8), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 130, 130), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 130, 130), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 65, 65), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 65, 65), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 33, 33), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 33, 33), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 17, 17), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 17, 17), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 17, 17), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 9, 9), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 9, 9), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 9, 9), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 190, 190), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 190, 190), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 95, 95), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 95, 95), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 48, 48), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 48, 48), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 24, 24), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 24, 24), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 24, 24), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 12, 12), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 12, 12), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 12, 12), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 150, 150), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 150, 150), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 75, 75), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 75, 75), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 38, 38), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 38, 38), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 19, 19), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 19, 19), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 816, 19, 19), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 816, 10, 10), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 10, 10), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 10, 10), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 96, 96), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 96, 96), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 48, 48), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 24, 24), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 24, 24), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 12, 12), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 12, 12), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 6, 6), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 6, 6), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 28, 28), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 12, 12), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 6, 6), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 3, 3), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 3, 3), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 80, 80), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 80, 80), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 40, 40), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 40, 40), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 20, 20), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 10, 10), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 10, 10), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 10, 10), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 5, 5), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 5, 5), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 5, 5), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)</td>
			<td>min : 0.0<br>max : 6.0</td>
		</tr>
	</tbody>
</table>
