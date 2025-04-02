<h1>Comprehensive Report on AvgPool2d Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of avgpool2d operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Avgpool2D Operation Details</th>
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
			<td rowspan="106">1</td>
			<td rowspan="106">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="106">222</td>
			<td>23</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_swin_swin_b_img_cls_torchvision</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>13</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_19_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
			<td>kernel_size : [1, 1]<br>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 10, 10), dtype=float32)</td>
			<td>kernel_size : [10, 10]<br>stride : [10, 10]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 35, 35), dtype=float32)</td>
			<td>kernel_size : [3, 3]<br>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>ceil_mode : False<br>count_include_pad : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 17, 17), dtype=float32)</td>
			<td>kernel_size : [3, 3]<br>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>ceil_mode : False<br>count_include_pad : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 8, 8), dtype=float32)</td>
			<td>kernel_size : [3, 3]<br>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>ceil_mode : False<br>count_include_pad : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 6, 6), dtype=float32)</td>
			<td>kernel_size : [1, 1]<br>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1920, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 8, 8), dtype=float32)</td>
			<td>kernel_size : [8, 8]<br>stride : [8, 8]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 10), dtype=float32)</td>
			<td>kernel_size : [10, 10]<br>stride : [10, 10]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2208, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 14, 14), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1664, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 10, 10), dtype=float32)</td>
			<td>kernel_size : [10, 10]<br>stride : [10, 10]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)</td>
			<td>kernel_size : [112, 112]<br>stride : [112, 112]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 112, 112), dtype=float32)</td>
			<td>kernel_size : [112, 112]<br>stride : [112, 112]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)</td>
			<td>kernel_size : [112, 112]<br>stride : [112, 112]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2688, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 8, 8), dtype=float32)</td>
			<td>kernel_size : [8, 8]<br>stride : [8, 8]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 9, 9), dtype=float32)</td>
			<td>kernel_size : [9, 9]<br>stride : [9, 9]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 12, 12), dtype=float32)</td>
			<td>kernel_size : [12, 12]<br>stride : [12, 12]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 10, 10), dtype=float32)</td>
			<td>kernel_size : [10, 10]<br>stride : [10, 10]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)</td>
			<td>kernel_size : [2, 2]<br>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 8, 8), dtype=float32)</td>
			<td>kernel_size : [8, 8]<br>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 6, 6), dtype=float32)</td>
			<td>kernel_size : [6, 6]<br>stride : [6, 6]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 3, 3), dtype=float32)</td>
			<td>kernel_size : [3, 3]<br>stride : [3, 3]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 5, 5), dtype=float32)</td>
			<td>kernel_size : [5, 5]<br>stride : [5, 5]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 96, 96), dtype=float32)</td>
			<td>kernel_size : [96, 96]<br>stride : [96, 96]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 48, 48), dtype=float32)</td>
			<td>kernel_size : [48, 48]<br>stride : [48, 48]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2904, 24, 24), dtype=float32)</td>
			<td>kernel_size : [24, 24]<br>stride : [24, 24]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7392, 12, 12), dtype=float32)</td>
			<td>kernel_size : [12, 12]<br>stride : [12, 12]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 696, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3712, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 104, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2520, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1008, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 912, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)</td>
			<td>kernel_size : [56, 56]<br>stride : [56, 56]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 216, 28, 28), dtype=float32)</td>
			<td>kernel_size : [28, 28]<br>stride : [28, 28]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1512, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 400, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1232, 14, 14), dtype=float32)</td>
			<td>kernel_size : [14, 14]<br>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3024, 7, 7), dtype=float32)</td>
			<td>kernel_size : [7, 7]<br>stride : [7, 7]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 1, 1), dtype=float32)</td>
			<td>kernel_size : [1, 1]<br>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
