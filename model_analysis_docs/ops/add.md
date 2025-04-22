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
			<td rowspan="63">1</td>
			<td rowspan="63">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="63">146</td>
			<td>7</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_57680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_148680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_168680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_217680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.0.branches.2.1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5412602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.0.fuse_layers.2.0.1.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6582602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.1.branches.1.0.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6342602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.1.fuse_layers.2.0.1.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8202602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_156358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage3.0.branches.0.3.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1242602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage3.1.branches.0.3.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2082602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage3.3.branches.0.2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3882602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage3.3.branches.2.1.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4482602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.0.branches.0.3.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4872602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.0.branches.1.3.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5172602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.0.branches.2.3.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_5532602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.1.branches.0.2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6102602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.1.branches.0.3.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6162602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.1.branches.2.3.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_6852602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.2.branches.1.3.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8112602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.2.fuse_layers.2.0.1.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9042602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_24208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_41208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=int64)</td>
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
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Constant, name=model.features.norm0.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1965, dtype=float32)</td>
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
			<td rowspan="59">2</td>
			<td rowspan="59">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="59">106</td>
			<td>8</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_80, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Constant, name=model.1.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_7214, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Constant, name=model.2.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_13214, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Constant, name=model.4.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25214, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_phi1_5_microsoft_phi_1_5_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi1_microsoft_phi_1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_60, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Constant, name=stage4.2.branches.1.2.bn2.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_8052602, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Constant, name=model.6.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_37214, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 522, 522), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_80, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 4, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi1_5_microsoft_phi_1_5_clm_hf</li><li>pt_phi1_microsoft_phi_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi1_5_microsoft_phi_1_5_seq_cls_hf</li><li>pt_phi1_microsoft_phi_1_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_70, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 11), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=model.0.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1214, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Constant, name=model.12.model.3.model.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73214, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 8, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 588, 588), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 596, 596), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_560, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_3445, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 522, 522), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 71, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 334), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 207, 207), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_90, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 207), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_90, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 107, 107), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 107, 107), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi4_microsoft_phi_4_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi4_microsoft_phi_4_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi4_microsoft_phi_4_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 29), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 39, 39), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_50, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=decoder.blocks.4.conv1.1.running_var, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_337910, dtype=float32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
