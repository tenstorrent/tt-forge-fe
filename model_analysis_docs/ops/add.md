<h1>Comprehensive Report on Add Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of add operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Add Operation Details</th>
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
			<td rowspan="61">1</td>
			<td rowspan="61">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="61">169</td>
			<td>9</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.0.branches.1.0.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4992602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.0.branches.1.1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5052602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage3.2.branches.1.1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3192602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.1.branches.0.1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6042602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage3.3.branches.1.2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4182602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.1.branches.0.2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6102602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.2.fuse_layers.2.0.1.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9042602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_156358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.6.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_133830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.6.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_136830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.7.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_145830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.8.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_154830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.9.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_163830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.10.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_172830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.11.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_181830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.12.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_190830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.13.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_199830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.14.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_208830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=layer3.15.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_217830, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_57680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_148680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_168680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_217680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_27802, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83802, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_247802, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1024), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=level4.tree2.tree1.tree1.tree2.tree1.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3401342, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=level4.tree2.tree1.tree2.tree1.tree1.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3611342, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=level4.tree2.tree2.tree1.tree1.tree2.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4121342, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=level4.tree2.tree2.tree1.tree2.tree1.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4241342, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=level4.tree2.tree2.tree1.tree2.tree2.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4331342, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=level4.tree2.tree2.tree2.tree1.tree1.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4451342, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_194452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.7.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1811238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.8.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1901238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.10.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2081238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.12.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2261238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.13.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2351238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.16.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2621238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.17.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2711238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.18.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2801238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.19.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2891238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.20.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2981238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.21.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3071238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.22.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3161238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.23.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3251238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.24.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3341238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.25.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3431238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.26.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3521238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.27.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3611238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Constant, name=backbones.ResNet152FPN.features.layer3.32.bn3.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4061238, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="3">2</td>
			<td rowspan="3">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td rowspan="3">11</td>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=int64)</td>
			<td></td>
		</tr>
	</tbody>
</table>
