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
			<td rowspan="103">1</td>
			<td rowspan="103">[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: resize2d</td>
			<td rowspan="103">196</td>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 15, 20), dtype=float32)</td>
			<td>sizes : [30, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 40), dtype=float32)</td>
			<td>sizes : [60, 80]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_yolox_yolox_s_obj_det_torchhub</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_320x320</li><li>pt_yolox_yolox_darknet_obj_det_torchhub</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 16, 16), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 32, 32), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 64, 64), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128, 128), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_yolox_yolox_l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 40, 40), dtype=float32)</td>
			<td>sizes : [80, 80]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_yolox_yolox_s_obj_det_torchhub</li><li>pt_yolox_yolox_darknet_obj_det_torchhub</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 40), dtype=float32)</td>
			<td>sizes : [80, 80]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 7, 7), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 32), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64, 64), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 128, 128), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolox_yolox_l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolox_yolox_m_obj_det_torchhub</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolox_yolox_m_obj_det_torchhub</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 40, 40), dtype=float32)</td>
			<td>sizes : [80, 80]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolox_yolox_x_obj_det_torchhub</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolox_yolox_x_obj_det_torchhub</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)</td>
			<td>sizes : [80, 80]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
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
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)</td>
			<td>sizes : [14, 14]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
			<td>sizes : [112, 112]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)</td>
			<td>sizes : [224, 224]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
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
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 80, 80), dtype=float32)</td>
			<td>sizes : [160, 160]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 15, 15), dtype=float32)</td>
			<td>sizes : [30, 30]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 30, 30), dtype=float32)</td>
			<td>sizes : [60, 60]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 10), dtype=float32)</td>
			<td>sizes : [20, 20]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)</td>
			<td>sizes : [20, 20]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 15, 15), dtype=float32)</td>
			<td>sizes : [30, 30]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 30), dtype=float32)</td>
			<td>sizes : [60, 60]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)</td>
			<td>sizes : [80, 80]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 15, 15), dtype=float32)</td>
			<td>sizes : [30, 30]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 30, 30), dtype=float32)</td>
			<td>sizes : [60, 60]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 10, 10), dtype=float32)</td>
			<td>sizes : [20, 20]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)</td>
			<td>sizes : [20, 20]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5m_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 10, 10), dtype=float32)</td>
			<td>sizes : [20, 20]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5m_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 20, 20), dtype=float32)</td>
			<td>sizes : [40, 40]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 15, 15), dtype=float32)</td>
			<td>sizes : [30, 30]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 30, 30), dtype=float32)</td>
			<td>sizes : [60, 60]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 15, 15), dtype=float32)</td>
			<td>sizes : [30, 30]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 30, 30), dtype=float32)</td>
			<td>sizes : [60, 60]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_tiny_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 13, 13), dtype=float32)</td>
			<td>sizes : [26, 26]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_tiny_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 26, 26), dtype=float32)</td>
			<td>sizes : [52, 52]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_nano_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 13, 13), dtype=float32)</td>
			<td>sizes : [26, 26]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_nano_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 26, 26), dtype=float32)</td>
			<td>sizes : [52, 52]<br>method : "nearest_neighbor"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
