<h1>Comprehensive Report on Sqrt Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of sqrt operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Sqrt Operation Details</th>
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
			<td rowspan="152">1</td>
			<td rowspan="152">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="152">1407</td>
			<td>113</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(64,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>109</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_fpn_base_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(256,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>103</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_resnet_50_img_cls_timm</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(128,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>97</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(512,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>77</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(32,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>69</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1024,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>49</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>35</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(192,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>35</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(16,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>29</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(96,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>28</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(160,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>26</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(144,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>25</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(24,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(320,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(384,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(672,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(960,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(224,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(48,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(240,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(768,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(112,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(72,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(480,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(576,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>13</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(40,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>13</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(80,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(288,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(448,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(120,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(336,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1536,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(36,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1152,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(352,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(56,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1632,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(896,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(864,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(88,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(60,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(18,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(728,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(720,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1056,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1248,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1344,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(416,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(544,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(608,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(640,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(704,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(736,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(800,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(832,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(928,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(992,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(8,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(432,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(528,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1392,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1440,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1920,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1792,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(272,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(208,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(12,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(20,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(100,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(92,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(200,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(184,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(816,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(912,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1008,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1728,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1824,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2016,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1120,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1184,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1216,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1312,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1376,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1408,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1472,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1504,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1568,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1600,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1664,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2688,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(232,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(44,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(176,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(30,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 588, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 54, 54), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 27, 27), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(624,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1104,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1200,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1296,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1488,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1584,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1680,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1776,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1872,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1968,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2064,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2112,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2160,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2208,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1696,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1760,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1856,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1888,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(136,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2904,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(7392,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(888,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(696,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3712,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(104,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(440,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2520,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(168,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(408,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(216,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1512,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(400,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1232,), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3024,), dtype=float32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
