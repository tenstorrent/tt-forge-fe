<h1>Comprehensive Report on Resize2d Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of resize2d operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Resize2D Operation Details</th>
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
			<td rowspan="40">1</td>
			<td rowspan="40">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="40">81</td>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 6, 20), dtype=float32)</td>
			<td>sizes : [12, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 12, 40), dtype=float32)</td>
			<td>sizes : [24, 80]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 24, 80), dtype=float32)</td>
			<td>sizes : [48, 160]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 48, 160), dtype=float32)</td>
			<td>sizes : [96, 320]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 96, 320), dtype=float32)</td>
			<td>sizes : [192, 640]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 32), dtype=float32)</td>
			<td>sizes : [20, 64]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 64), dtype=float32)</td>
			<td>sizes : [40, 128]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 128), dtype=float32)</td>
			<td>sizes : [80, 256]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 80, 256), dtype=float32)</td>
			<td>sizes : [160, 512]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 160, 512), dtype=float32)</td>
			<td>sizes : [320, 1024]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 27, 27), dtype=float32)</td>
			<td>sizes : [27, 27]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 27, 27), dtype=float32)</td>
			<td>sizes : [27, 27]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 8), dtype=float32)</td>
			<td>sizes : [16, 16]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)</td>
			<td>sizes : [64, 64]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)</td>
			<td>sizes : [7, 7]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>sizes : [112, 112]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
			<td>sizes : [224, 224]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)</td>
			<td>sizes : [20, 20]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
