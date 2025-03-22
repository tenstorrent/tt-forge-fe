<h1>Comprehensive Report on Squeeze Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of squeeze operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Squeeze Operation Details</th>
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
			<td rowspan="129">1</td>
			<td rowspan="129">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="129">422</td>
			<td>31</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>31</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>ResNetForImageClassification</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_xception_xception_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>13</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 196, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 16384, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 1, 64), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 64), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16384, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 4096, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 1024, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1024, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 256, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 256, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 25088, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 25088, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1), dtype=int32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 2048, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1024, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 196, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 32, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 1024), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1024), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024, 1), dtype=float32)</td>
			<td>dim : -4</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1), dtype=float32)</td>
			<td>dim : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024, 72), dtype=float32)</td>
			<td>dim : -4</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 72), dtype=float32)</td>
			<td>dim : -3</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_mnist_base_img_cls_github</li><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9216, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li><li>pt_mnist_base_img_cls_github</li><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9216, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1920, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1920, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16384, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 1, 32), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 32), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16384, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 4096, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4096, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 1024, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 1024, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 256, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 576, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 768, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 768, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 512), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 1, 512), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 322), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3025, 1, 322), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 261), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 1, 261), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 128), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3072, 1, 128), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 96, 54, 54), dtype=float32)</td>
			<td>dim : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 27, 27), dtype=float32)</td>
			<td>dim : -4</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 196, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 196, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2208, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2208, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1664, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1664, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla34_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1000, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla34_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1000, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 19200, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 1, 64), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 1, 64), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 19200, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4800, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 4800, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 1200, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1200, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 300, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 300, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 480, 640), dtype=float32)</td>
			<td>dim : -3</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1), dtype=float32)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7392, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7392, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3712, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3712, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2520, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2520, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1008, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1008, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 912, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 912, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1512, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1512, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 400, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 400, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3024, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3024, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 3136, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 1, 1), dtype=float32)</td>
			<td>dim : -2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
	</tbody>
</table>
