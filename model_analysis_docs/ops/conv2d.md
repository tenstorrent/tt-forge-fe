<h1>Comprehensive Report on Conv2d Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of conv2d operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Conv2D Operation Details</th>
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
			<td rowspan="1987">1</td>
			<td rowspan="1987">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="1987">4956</td>
			<td>34</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>32</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>30</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>28</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>27</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>25</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>25</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>24</td>
			<td><ul><li>pt_resnet_50_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>24</td>
			<td><ul><li>pt_resnet_50_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>23</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>23</td>
			<td><ul><li>pt_resnet_50_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>22</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>22</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>21</td>
			<td><ul><li>pt_resnet_50_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>21</td>
			<td><ul><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>20</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>19</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>17</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnet_50_img_cls_timm</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>15</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>13</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 3, 7, 7), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 3, 16, 16), dtype=float32)</td>
			<td>stride : [16, 16]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 8, 8), dtype=float32)</td>
			<td>stride : [8, 8]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 4, 4), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 512<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 320, 2, 2), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1280<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 320, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2048<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 2048, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>8</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_resnet_50_img_cls_timm</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnet_resnet101_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 2048, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2560, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 2560, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1000, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 4, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 8, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 192, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 48, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 48, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 24, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 48, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 24, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 12, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 24, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 12, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 6, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 12, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 8, 22), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 42), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 42), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 26, 82), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 26, 82), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 50, 162), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 50, 162), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 98, 322), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 96, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 98, 322), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 194, 642), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 194, 642), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 8, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 72<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(168, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 168, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 168, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 480, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 36, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 36, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 36, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 18, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 36, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 72, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 18, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 72, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 144, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(126, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 18, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 18, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 18, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 36, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 36, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 36, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 18, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w18_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 18, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 160, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1472, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 1472, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 768, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1728, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 1728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 768, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 224, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1888, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1888, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2144, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 2144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(728, 728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(728, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 728<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 728, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1024<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1536, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1536<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception41_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 3, 16, 16), dtype=float32)</td>
			<td>stride : [16, 16]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 352, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 416, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 416, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 352, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 416, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 416, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 544, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 544, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 608, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 608, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 640, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 704, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 704, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 736, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 736, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 800, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 800, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 832, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 832, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 864, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 864, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 928, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 928, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 992, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 992, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 928, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 928, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 992, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 992, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 832, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 2048, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(6, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 6, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(28, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 28, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_xception_xception_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 299, 299), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 299, 299), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 384<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 576<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 2048, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(728, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(728, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 728<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(728, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 728<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(728, 728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1024<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 640, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 704, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 704, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 736, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 736, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 800, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 800, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 864, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 864, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2816, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 2816, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 640, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1000, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1280, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1152, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1152<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1152<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 12, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(12, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 12<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(36, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 36<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 20, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(20, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 20<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(60, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 60<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 120<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 40<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(40, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 40<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(100, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 100, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(100, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 100<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 200, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(92, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 92, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(92, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 92<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 184, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(56, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 56<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 80<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(80, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 80<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 112<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_in1k_img_cls_timm</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 80, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 149, 149), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 147, 147), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 147, 147), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 73, 73), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 73, 73), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 73, 73), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 1, 7), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 3, 0, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 73, 73), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 7, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [3, 0, 3, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 71, 71), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 96, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 384, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 192, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 35, 35), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 224, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 192, 1, 7), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 3, 0, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 224, 7, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [3, 0, 3, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 7, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [3, 0, 3, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 224, 7, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [3, 0, 3, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 224, 1, 7), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 3, 0, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 7), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 3, 0, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 256, 7, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [3, 0, 3, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 320, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 384, 1, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 384, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 384, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 448, 1, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 512<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1024<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 576<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 2048, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 320, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 12, 34), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 22, 66), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 22, 66), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 42, 130), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 42, 130), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 82, 258), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 82, 258), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 162, 514), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 96, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 162, 514), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 322, 1026), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 322, 1026), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 2048, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 2048, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 2048, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3072, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 3072, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(150, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 3, 4, 4), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 384, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1088, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1184, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1216, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1216, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1248, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1088, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1120, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1184, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1216, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1216, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1248, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1280, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1312, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1312, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1344, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1376, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1376, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1408, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1408, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1440, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1472, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1472, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1504, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1504, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1568, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1568, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1600, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1600, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 544, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 544, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 608, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 608, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 4, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 8, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2304, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 2304, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 2, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 2, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 2, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 2, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 4, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 4, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 8, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 8, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1000, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 4, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 8, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 8, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 4, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(10, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 10, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(20, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 20, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1152, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(6, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 6, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(14, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 14, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(68, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 68, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1632, 68, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2688, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 2688, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2688, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 160, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w64_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w64_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 48, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 96, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 96, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 96, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 192, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 384, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 96, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w48_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w48_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(112, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w32_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w32_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 40, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 40, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 80, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 40, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 80, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 160, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 40, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 320, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(280, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 40, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 40, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 40, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 80, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w40_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w40_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 44, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 44, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 88, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 88, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 88, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 44, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(176, 88, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(176, 176, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 176, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 176, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(176, 44, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(352, 176, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(352, 352, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(308, 352, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(176, 44, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(44, 44, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(352, 44, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(352, 88, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 352, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 352, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 176, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 176, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 88, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 88, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 44, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w44_pose_estimation_timm</li><li>pt_hrnet_hrnetv2_w44_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 44, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 44, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 30, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 30, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 60, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 60, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 60, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 30, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 60, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 120, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(60, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 30, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 120, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 240, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(210, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 30, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 30, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 30, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 60, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 60, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 60, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 60, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 30, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnetv2_w30_pose_estimation_osmr</li><li>pt_hrnet_hrnet_w30_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 30, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</li><li>pt_mobilenet_v1_basic_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 512<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 72<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 72<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(88, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 88<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 88, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 120<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 288<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 576<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 120<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(200, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 200<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 200, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(184, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 184<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 184, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 184, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 3<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 3<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 56, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 224, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 56, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(150, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 7, 7), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 8, 8), dtype=float32)</td>
			<td>stride : [8, 8]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 4, 4), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 160, 2, 2), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 640, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(640, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 640<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 160, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1024<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(160, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 160<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 736, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 736, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 1088, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(224, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 224<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1440, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xception_xception41_img_cls_timm</li><li>pt_xception_xception65_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(728, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xception_xception71_img_cls_timm</li><li>pt_xception_xception71_tf_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(728, 728, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [1, 0, 1, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 336, 336), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 3, 14, 14), dtype=float32)</td>
			<td>stride : [14, 14]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 3, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(3072, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3072, 1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 11, 11), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 27, 27), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 64, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 192, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 3, 11, 11), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 27, 27), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 96, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 384, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 3, 16, 16), dtype=float32)</td>
			<td>stride : [16, 16]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 3, 16, 16), dtype=float32)</td>
			<td>stride : [16, 16]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 3, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 96, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 432, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 624, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 624, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 432, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 624, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 624, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 816, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 816, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 864, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 864, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 912, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 912, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1008, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1008, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1104, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1104, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1200, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1248, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1296, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1296, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1344, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1440, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1440, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1488, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1488, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1584, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1584, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1680, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1680, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1728, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1776, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1776, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1824, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1824, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1872, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1872, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1920, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1920, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1968, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1968, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2016, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2064, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2064, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 2112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1104, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1104, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1200, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1248, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1296, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1296, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1344, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1440, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1488, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1488, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1584, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1584, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1680, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1680, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1728, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1776, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1776, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1824, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1824, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1872, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1872, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1920, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1920, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1968, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1968, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2016, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2064, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2064, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 2160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1280, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1312, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1312, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1344, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1376, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1376, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1408, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1408, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1440, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1440, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1472, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1472, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1504, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1504, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1536, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1568, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1568, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1600, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1600, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1664, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1664, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1696, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1696, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1728, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1760, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1760, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 1792, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1664, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1664, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1696, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1696, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1728, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1760, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1760, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1792, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1824, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1824, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1856, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1856, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet201_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1888, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1888, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 1, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet169_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(640, 1280, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2560, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 2560, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla169_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3328, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 3328, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 4, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 2, 2]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 56, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(272, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 272, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1632, 272, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1632<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(272, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1632<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2688, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2688, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2688, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2688<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2688, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 2688, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1792, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 56, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(272, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 272, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1632, 272, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1632<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(272, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1632<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2688, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2688, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2688, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2688<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2688, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 2688, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1792, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 240, 240), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 120, 120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 120, 120), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 120, 120), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 120, 120), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 60, 60), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 60, 60), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 60, 60), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 60, 60), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 60, 60), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 2, 2]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 30, 30), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 40, 30, 30), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 40, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 30, 30), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 30, 30), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(40, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 30, 30), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 15, 15), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1152, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1152<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1152, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1152<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1152, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 1152, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 260, 260), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 130, 130), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 130, 130), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 130, 130), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 130, 130), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 65, 65), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 65, 65), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 65, 65), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 65, 65), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 65, 65), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 33, 33), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 33, 33), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 33, 33), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 288<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 33, 33), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 33, 33), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 288<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 88, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528, 88, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(528, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 528<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(88, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(528, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 528<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 720<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 720<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 17, 17), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 9, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 9, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1248, 208, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 9, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1248, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1248<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 9, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 1248, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 9, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1248, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1248<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1248, 9, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(352, 1248, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 352, 9, 9), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 352, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 380, 380), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 190, 190), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 190, 190), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 190, 190), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 190, 190), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 95, 95), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 95, 95), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 95, 95), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 95, 95), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 95, 95), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 56, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(272, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 272, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1632, 272, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1632<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(272, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1632, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1632<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1632, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 1632, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 300, 300), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(32, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 32<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 150, 150), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 288<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 288<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 576<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 576<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(136, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 136, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(816, 136, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 816, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(816, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 816<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 816, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(816, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 816<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 816, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(136, 816, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 816, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 816, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392, 232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1392, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1392<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1392, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1392<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 2048, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 1, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 72<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(72, 1, 5, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 72<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 1, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 120<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(120, 1, 5, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 120<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 1, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(240, 1, 5, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 240<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(200, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200, 1, 1, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 200<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 200, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(200, 1, 5, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 200<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(184, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184, 1, 1, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 184<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 184, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(184, 1, 5, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 184<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(480, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 1, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(480, 1, 5, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 480<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 1, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 1, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 2, 0, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 0, 2, 0]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 480, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 8, 8), dtype=float32)</td>
			<td>stride : [8, 8]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 4, 4), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 512<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 320, 2, 2), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1280<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 320, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2048<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 15, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 30, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 60, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 120, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 480, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 480, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(176, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 96, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(304, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 96, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 480, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(296, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 112, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(280, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 144, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 160, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(448, 832, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 160, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(624, 832, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 192, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 48, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 512, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mnist_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mnist_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 26, 26), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 512<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 192, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 384<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 384<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 768<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 2<br>groups : 384<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 2<br>groups : 576<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [4, 4, 4, 4]<br>dilation : 4<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(21, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 56, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(336, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 336<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 576<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(24, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(288, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 288<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 432<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 432<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 432, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 432, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(720, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 720<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1792, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 227, 227), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.0.weight, dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 27, 27), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.3.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.6.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.8.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 13, 13), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.10.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 3<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 3<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 192, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 512, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 17<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1088, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 1088, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 17<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1088, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(272, 1088, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_facebook_regnet_y_040_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 272, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1088, 272, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 384, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 192, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 192, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 192, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528, 264, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528, 264, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(132, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 132, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(528, 132, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 528, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 264, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(132, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 132, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 132, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 264, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(264, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 264, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1056, 264, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2904, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1056, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2904, 1056, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2904, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2904, 264, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 11<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2904, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(264, 2904, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 264, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2904, 264, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2904, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2904, 2904, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2904, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2904, 264, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 11<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2904, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(726, 2904, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 726, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2904, 726, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2904, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(7392, 2904, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2904, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(7392, 2904, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7392, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(7392, 264, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 28<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7392, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(726, 7392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 726, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(7392, 726, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_128gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7392, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(7392, 7392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 48, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 5<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 12, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 5<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 30, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 120, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 14<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(30, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 30, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 30, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 14<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(84, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 84, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 84, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(888, 336, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(888, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(888, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 37<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(84, 888, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 84, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(888, 84, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(888, 888, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(888, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 37<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 888, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(222, 888, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 222, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(888, 222, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 232, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 232, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(58, 232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 58, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(232, 58, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(696, 232, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 232, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(696, 232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 696, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(696, 232, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 3<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 696, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(58, 696, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 58, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(696, 58, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 696, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(696, 696, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 696, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(696, 232, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 3<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 696, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(174, 696, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 174, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(696, 174, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 696, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392, 696, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 696, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392, 696, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392, 232, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 6<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(174, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 174, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392, 174, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392, 232, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 6<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(348, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 348, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1392, 348, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3712, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1392, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3712, 1392, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3712, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3712, 232, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3712, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(348, 3712, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 348, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3712, 348, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3712, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3712, 3712, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 8, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 6<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(104, 48, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(104, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 104, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(104, 8, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 13<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 104, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(12, 104, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(104, 12, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 104, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(104, 104, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 104, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(104, 8, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 13<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 104, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(26, 104, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 26, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(104, 26, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 104, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 104, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 104, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 104, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 8, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 26<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(26, 208, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 26, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 26, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 208, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 8, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 26<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(52, 208, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 52, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(208, 52, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(440, 208, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 208, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(440, 208, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(440, 8, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 55<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(52, 440, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 52, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(440, 52, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(440, 440, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(440, 8, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 55<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 440, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(110, 440, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 110, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(440, 110, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 80, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 80, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 80, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 120, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 120, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 240, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 120, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 6<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 120, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 6<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1920, 720, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1920, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1920, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1920, 120, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1920, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1920, 1920, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 168, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 168, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 336, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 168, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 168, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1344, 672, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1344, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1344, 168, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1344, 1344, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1344, 168, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2520, 1344, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1344, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2520, 1344, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2520, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2520, 168, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 15<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_32gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2520, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2520, 2520, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 32, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 48, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 48, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432, 192, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 9<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432, 432, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432, 48, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 9<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1008, 432, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1008, 432, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1008, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1008, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 21<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1008, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1008, 1008, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1008, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1008, 48, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 21<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 512, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 7<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 7<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 896, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(168, 72, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(168, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 168, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(168, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 7<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 168, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(168, 168, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 168, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(168, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 7<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 168, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(408, 168, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 168, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(408, 168, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 408, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(408, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 17<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 408, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(408, 408, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 408, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(408, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 17<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 408, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(912, 408, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 408, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(912, 408, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 912, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(912, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 38<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 912, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(912, 912, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_1_6gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 912, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(912, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 38<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 18<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 18<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 288, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 42<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 672, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(672, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 42<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 56, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 56, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 56, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 56, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 8<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 448, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 56, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 56, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 16<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(896, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2016, 896, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 896, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2016, 896, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2016, 56, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 36<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 2016, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2016, 224, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_8gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2016, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2016, 2016, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 9<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 9<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 36, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 144, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 20<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(36, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 36, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 36, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 20<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(784, 320, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(784, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(784, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 49<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 784, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(784, 80, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(784, 784, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(784, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 49<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(196, 784, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_800mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(784, 196, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 18, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(216, 72, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(216, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 216, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(216, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 9<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 216, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(18, 216, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 18, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(216, 18, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 216, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(216, 216, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 216, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(216, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 9<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 216, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(54, 216, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 54, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(216, 54, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 216, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 216, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 216, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 216, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(54, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 54, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 54, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 24, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 24<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1512, 576, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1512, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1512, 24, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 63<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 1512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1512, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_3_2gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1512, 1512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 10<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 10<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(400, 160, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(400, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 400, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(400, 16, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 25<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 400, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(400, 400, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_x_400mf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 400, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(400, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 25<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 112, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(224, 112, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 2<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 112, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(448, 112, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1232, 448, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1232, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1232, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1232, 112, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 11<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1232, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 1232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1232, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1232, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1232, 1232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1232, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1232, 112, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 11<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1232, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(308, 1232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 308, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1232, 308, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1232, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3024, 1232, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1232, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3024, 1232, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3024, 112, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 27<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(308, 3024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 308, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3024, 308, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_regnet_regnet_y_16gf_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3024, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(3024, 3024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 300, 300), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 75, 75), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1024, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(324, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 19, 19), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(486, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(486, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(486, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(324, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(324, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 3, 4, 4), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(19, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_carvana_base_img_seg_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3072, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 3072, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 768, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 384, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1, 16, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096, 512, 7, 7), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096, 4096, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 448, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(80, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 528, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 96, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 736, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 736, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 384, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 112, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 944, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 944, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 149, 149), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 147, 147), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 147, 147), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 147, 147), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 147, 147), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 147, 147), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 74, 74), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 74, 74), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 74, 74), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 74, 74), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 74, 74), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 37, 37), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 37, 37), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(728, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 37, 37), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(728, 256, 1, 1), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 37, 37), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(728, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 728<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 37, 37), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(728, 728, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.0.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.1.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.2.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.2.m.0.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.2.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.2.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.2.cv3.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.3.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.4.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.4.m.0.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.4.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.4.m.1.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.4.m.1.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.4.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.4.cv3.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.5.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.0.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.1.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.1.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.2.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.2.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.cv3.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.7.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.8.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.8.m.0.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.8.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.8.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.8.cv3.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.9.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.9.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.10.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.m.0.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.cv3.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.14.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.17.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.17.m.0.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.17.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.17.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.17.cv3.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.24.m.0.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.18.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.m.0.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.cv3.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.24.m.1.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.21.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.23.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.23.m.0.cv1.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.23.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.23.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.23.cv3.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.24.m.2.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
