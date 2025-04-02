<h1>Comprehensive Report on Unsqueeze Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of unsqueeze operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Unsqueeze Operation Details</th>
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
			<td rowspan="577">1</td>
			<td rowspan="577">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="577">4036</td>
			<td>139</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>ResNetForImageClassification</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_mnist_base_img_cls_github</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>133</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_fpn_base_img_cls_torchvision</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_alexnet_base_img_cls_osmr</li><li>ResNetForImageClassification</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>126</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>ResNetForImageClassification</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_resnet_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(128, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>116</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>ResNetForImageClassification</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(512, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>113</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>ResNetForImageClassification</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(64,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>109</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_fpn_base_img_cls_torchvision</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>ResNetForImageClassification</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(256,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>103</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>ResNetForImageClassification</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_resnet_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(128,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>97</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>ResNetForImageClassification</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(512,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>97</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_mnist_base_img_cls_github</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>77</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(32,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>76</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>71</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_vgg_19_obj_det_hf</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_mlp_mixer_base_img_cls_github</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>69</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1024,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>59</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2048, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>51</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>49</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(2048,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>48</td>
			<td><ul><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_autoencoder_conv_img_enc_github</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>44</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_vgg_19_obj_det_hf</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_mnist_base_img_cls_github</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>43</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.rotary_emb.inv_freq, dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>41</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_vgg_19_obj_det_hf</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>38</td>
			<td><ul><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(192, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>35</td>
			<td><ul><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(192,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>35</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>33</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(96, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>31</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_mlp_mixer_base_img_cls_github</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>30</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(160, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>30</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(320, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>29</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(96,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>29</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1280, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>29</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>28</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>28</td>
			<td><ul><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(160,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>27</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(144, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>27</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>26</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(768, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>26</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(144,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>25</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(24,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>24</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_mnist_base_img_cls_github</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>23</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>23</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 128), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>23</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(384, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(320,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(384,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(672,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(672, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(960,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(960, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(224,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(224, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(112, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(48,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(240,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(240, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(768,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1280,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(8,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(40, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(112,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(72,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(72, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_autoencoder_conv_img_enc_github</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(16,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_autoencoder_conv_img_enc_github</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_unet_base_img_seg_torchhub</li><li>pt_autoencoder_conv_img_enc_github</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(480,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(480, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(576,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(576, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(80, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>14</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(36, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>13</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(40,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>13</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(80,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(288,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(288, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(448,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(448, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(120,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(120, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1000,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1000, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(320,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(336,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(336, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(720, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(240,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(672,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(56, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li><li>pt_gemma_google_gemma_2b_text_gen_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.layers.0.self_attn.rotary_emb.inv_freq, dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1536,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1536, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(144,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ssd300_resnet50_base_img_cls_torchhub</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(24,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(36,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(72,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(120,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_fuyu_adept_fuyu_8b_qa_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1152,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1152, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(352,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(352, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(480,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(48,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(56,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(960,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 32), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1632,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1632, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(640, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(896,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(896, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(168, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(18, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(36,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 4, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(864,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(864, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(20,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(20, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(88,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(88, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(60,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(60, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(168,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(18,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(720,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(150,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(150, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(728,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(728, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(720,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1056,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1056, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1248,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1248, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1344,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1344, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(416,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(416, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(544,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(544, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(608,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(608, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(640,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(704,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(704, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(736,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(736, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(800,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(800, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(832,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(832, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(928,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(928, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(992,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(992, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(6,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(28,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(28, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(12,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(40,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(272, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(112,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(8,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 13, 13), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 13), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.model.rotary_emb.inv_freq, dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 256), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 10, 256), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1, 1024), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(512, 1024), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Parameter, shape=(4,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(432,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(432, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(528,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(528, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1392,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1392, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1440,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1440, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1920,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1920, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1088,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1088, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1792,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1792, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(336,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(272,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(208,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(208, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(12,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(20,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(100,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(100, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(92,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(92, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(200,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(200, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(184,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(184, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(30, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(576,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 49, 49), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 49, 49), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 49, 49), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(24, 49, 49), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 64), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 256), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 256), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 64), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 64), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 4, 64), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 32), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 32), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 32), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 64), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 35, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 39, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 1, 1), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 61, 61), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 1, 1), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 61, 61), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(816,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(816, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(912,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(912, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1008,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1008, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1728,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1728, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1824,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1824, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2016,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2016, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1120,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1120, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1184,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1184, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1216,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1216, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1312,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1312, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1376,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1376, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1408,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1408, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1472,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1472, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1504,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1504, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1568,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1568, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1600,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1600, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1664,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1664, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(10,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(10, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1152,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(14,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(14, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(68,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(68, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(1632,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(2688,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(2688, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(2688,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(232,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(232, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(44,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(44, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(176,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(176, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(30,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(288,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(224,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(56,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(448,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(160,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(640,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 6), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 7), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 588, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596), dtype=uint1)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 204), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 204), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 201), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 14), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 9), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.rotary_emb.inv_freq, dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=fuyu_model.language_model.model.rotary_emb.inv_freq, dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 32), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 256), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 256), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 256), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 207, 256), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 107, 256), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 32, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(50176, 256), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 768), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(256, 1280), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 2048), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 96), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 96), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 96), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 64), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 64), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 64), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 39, 64), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 39, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 13, 128), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 13, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 29, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 192, 1), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 192, 1), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(3072, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3072, 128), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 768, 1), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1, 1), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 61, 61), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 1, 1), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(6, 61, 61), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 197, 197), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 197, 197), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(624,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(624, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1104,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1104, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1200,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1200, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1296,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1296, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1488,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1488, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1584,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1584, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1680,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1680, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1776,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1776, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1872,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1872, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1968,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1968, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2064,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2064, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2112,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2112, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2160,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2160, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2208,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2208, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1696,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1696, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1760,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1760, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1856,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1856, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1888,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1888, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(136,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(136, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 40), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 80), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 160), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 224), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(21,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(21, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(3,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Constant, name=features.0.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Constant, name=features.3.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Constant, name=features.6.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Constant, name=features.8.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Constant, name=features.10.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1088,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(272,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(528,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(132,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(132, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1056,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(264,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(264, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2904,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2904, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(2904,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(726,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(726, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(7392,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(7392, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(7392,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(30,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(84,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(84, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(888,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(888, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(888,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(222,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(222, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(232,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(58,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(58, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(696,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(696, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(696,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(174,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(174, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1392,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(348,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(348, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3712,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3712, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(3712,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(104,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(104, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(104,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(26,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(26, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(208,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(52,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(52, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(440,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(440, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(440,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(110,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(110, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2520,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(2520, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(168,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(408,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(408, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(896,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(2016,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(80,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(784, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(784,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(196,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(196, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(18,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(216,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(216, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(216,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(54,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(54, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1512,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1512, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1512,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(400,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(400, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1232,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1232, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(1232,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(308,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(308, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3024,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3024, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(3024,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(324,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(324, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Parameter, shape=(486,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(486, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 49, 49), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 49, 49), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 49, 49), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 49, 49), dtype=float32)</td>
			<td>dim : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Parameter, shape=(19,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(19, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(4096,), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(4096, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.0.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.2.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.2.m.0.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.2.m.0.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.2.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.2.cv3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.4.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.4.m.0.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.4.m.0.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.4.m.1.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.4.m.1.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.4.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.4.cv3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.5.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.m.0.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.m.0.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.m.1.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.m.1.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.m.2.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.m.2.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.6.cv3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.7.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.8.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.8.m.0.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.8.m.0.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.8.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.8.cv3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.9.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.9.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.10.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.13.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.13.m.0.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.13.m.0.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.13.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.13.cv3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.14.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.17.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.17.m.0.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.17.m.0.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.17.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.17.cv3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.24.m.0.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(255, 1), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.18.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.20.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.20.m.0.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.20.m.0.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.20.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.20.cv3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.24.m.1.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.21.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.23.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.23.m.0.cv1.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.23.m.0.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.23.cv2.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.23.cv3.conv.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Constant, name=model.model.model.24.m.2.bias, dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
	</tbody>
</table>
