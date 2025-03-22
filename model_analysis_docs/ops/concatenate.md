<h1>Comprehensive Report on Concatenate Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of concatenate operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Concatenate Operation Details</th>
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
			<td rowspan="623">1</td>
			<td rowspan="623">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="623">1110</td>
			<td>7</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 35, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla34_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 12, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 12, 40), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 24, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 24, 80), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 48, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 48, 160), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 96, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 96, 320), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 192, 640), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 4, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 4, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 4, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(18, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 144, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(36, 18, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 18, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(72, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 144, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 39, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 256, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 256, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2, 35, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 196, 768), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 10, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 10, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 10, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 29, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 20, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 73, 73), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 73, 73), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 71, 71), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 71, 71), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 35, 35), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(96, 384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 384, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 35, 35), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 384, 17, 17), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 1024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1024, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 17, 17), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 320, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1024, 8, 8), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 1536, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 1536, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 1536, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 8, 8), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 8, 8), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 20, 64), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 40, 128), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 80, 256), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 160, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 160, 512), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 320, 1024), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 768, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 768, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 768, 128, 128), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 53, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 56, 96), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 53, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 3, 96), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 53, 56, 96), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 3, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 53, 96), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 96), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 25, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 28, 192), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 25, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 3, 192), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 25, 28, 192), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 3, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 25, 192), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 192), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 14, 384), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 11, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 3, 384), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 11, 14, 384), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 3, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 11, 384), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 384), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 6, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 10, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 207, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 207, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 107, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 256, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 256, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 4, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 4, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 4, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 16), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 256, 16), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 256, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 11, 16), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 11, 16), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 11, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 16), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 12, 16), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 12, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 256, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 29, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 35, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 35, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 35, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 35, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 35, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 35, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 35, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 35, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 39, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2, 39, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 196, 1024), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 512, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(48, 384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(96, 48, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 48, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 48, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(48, 48, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 48, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 48, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 384, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 16, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 128, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(32, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 256, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(40, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 320, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(80, 40, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 40, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 40, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(80, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(40, 40, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 40, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 40, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(160, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 320, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(44, 352, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 352, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(176, 352, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(44, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(88, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(176, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(88, 44, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 44, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 44, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(88, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(44, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(44, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(44, 44, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 44, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 44, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(44, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(44, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(88, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(176, 352, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 352, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 352, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(176, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(88, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(44, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(30, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 240, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(30, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(60, 30, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 30, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 30, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(60, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(30, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(30, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(30, 30, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 30, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 30, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(30, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(30, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(120, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 240, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(30, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 128, 128), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 224, 224), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 588, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 588, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 588, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 588, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 39, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 576, 1024), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 596, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 596, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 596, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 10, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 71, 6, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 71, 6, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 6, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 6, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 334, 16), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 334, 16), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 334, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 207, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 207, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 7, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 7, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 207, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 207, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 107, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 107, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 107, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 107, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 107, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 107, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 4, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 24, 4, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 32, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 24, 32, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 32, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 32, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 32, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 128, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 128, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 8, 128, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 50176, 256), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3025, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 5, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 5, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 5, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 13, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 13, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 13, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 48), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 29, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 6, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 35, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 35, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2, 35, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 39, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 39, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 29, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 29, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 39, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 39, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 39, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 39, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 39, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 39, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2, 39, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 39, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 39, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 39, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 39, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 29, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2, 29, 32), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 13, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 13, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 13, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 13, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 13, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 29, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 29, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 29, 64), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(729, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3, 12), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(729, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3, 16), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 196, 384), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 196, 192), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 416, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 416, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 544, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 608, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 704, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 736, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 800, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 832, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 864, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 928, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 992, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 544, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 608, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 704, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 736, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 800, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 864, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 928, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 992, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 30, 40), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 60, 80), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 224, 224), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(64, 192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 192, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 256, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(192, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 480, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 208, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(160, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 512, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 224, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 512, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(112, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 512, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(112, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 528, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 528, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 528, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 832, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 832, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 832, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 320, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 832, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 832, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 832, 1, 1), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 384, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)</td>
			<td>axis : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 5776), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 2166), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 600), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 36), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4, 4), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 81, 5776), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 81, 2166), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 81, 600), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 81, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 81, 36), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 81, 4), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 53, 56, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 56, 128), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 53, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 3, 128), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 56, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 53, 56, 128), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 3, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 56, 53, 128), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 28, 128), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 25, 28, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 28, 256), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 25, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 3, 256), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 28, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 25, 28, 256), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 3, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 28, 25, 256), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 14, 256), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 14, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 3, 14, 512), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 11, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 3, 512), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 14, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 11, 14, 512), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 3, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 14, 11, 512), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 7, 7, 512), dtype=float32)</td>
			<td>axis : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 32, 32), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 16, 80, 80), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32, 40, 40), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=const_10, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_30, dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=const_60, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70, dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=const_120, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_130, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_140, dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=const_170, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_180, dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=const_230, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_240, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_250, dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=const_280, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_290, dtype=float32)</td>
			<td>axis : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 85), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1200, 85), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 300, 85), dtype=float32)</td>
			<td>axis : -2</td>
		</tr>
	</tbody>
</table>
