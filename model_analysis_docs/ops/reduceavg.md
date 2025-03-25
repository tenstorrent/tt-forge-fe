<h1>Comprehensive Report on ReduceAvg Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of reduceavg operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Reduceavg Operation Details</th>
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
			<td rowspan="51">1</td>
			<td rowspan="51">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="51">85</td>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 1, 14), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 14), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 7), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 768), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 28), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 28), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 7), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 512), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 768), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 4096), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 2048), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 2304), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1024), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 896), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 896), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 1024), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 768), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 1024), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1, 112), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 1, 56), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 56), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 28), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 28), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 14), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 1, 7), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 256), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 56), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 28), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 14), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)</td>
			<td>dim : -2<br>keep_dim : True</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 7), dtype=float32)</td>
			<td>dim : -1<br>keep_dim : True</td>
		</tr>
	</tbody>
</table>
