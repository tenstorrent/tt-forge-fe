<h1>Comprehensive Report on MaxPool2d Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of maxpool2d operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Maxpool2D Operation Details</th>
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
			<td rowspan="61">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="61">249</td>
			<td>23</td>
			<td><ul><li>pt_resnet_50_img_cls_timm</li><li>pt_densenet_densenet169_img_cls_torchvision</li><li>pt_densenet_densenet201_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_timm</li><li>pt_unet_qubvel_img_seg_torchhub</li><li>pt_densenet_densenet121_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_torchvision</li><li>pt_densenet_densenet121_hf_xray_img_cls_torchvision</li><li>pt_resnet_resnet18_img_cls_torchvision</li><li>pt_resnext_resnext50_32x4d_img_cls_osmr</li><li>pt_resnext_resnext26_32x4d_img_cls_osmr</li><li>pt_resnet_resnet50_img_cls_torchvision</li><li>pt_resnext_resnext14_32x4d_img_cls_osmr</li><li>pt_resnet_resnet101_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet101_2_img_cls_timm</li><li>pt_resnext_resnext101_64x4d_img_cls_osmr</li><li>pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</li><li>pt_resnext_resnext50_32x4d_img_cls_torchhub</li><li>ResNetForImageClassification</li><li>pt_resnet_resnet34_img_cls_torchvision</li><li>pt_resnet_resnet152_img_cls_torchvision</li><li>pt_wideresnet_wide_resnet50_2_img_cls_torchvision</li><li>pt_resnext_resnext101_32x8d_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>22</td>
			<td><ul><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_vgg_19_obj_det_hf</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_dla_dla102x2_visual_bb_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 224, 224), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 112, 112), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>18</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_vgg13_bn_img_cls_torchvision</li><li>pt_vgg_vgg11_img_cls_torchvision</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_img_cls_torchvision</li><li>pt_unet_carvana_base_img_seg_github</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg13_img_cls_torchvision</li><li>pt_vgg_vgg11_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_img_cls_torchvision</li><li>pt_vgg_vgg16_bn_img_cls_torchvision</li><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 112, 112), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_v1_vovnet39_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet_v1_57_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</li><li>pt_vovnet_vovnet57_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla60_visual_bb_torchvision</li><li>pt_dla_dla60x_visual_bb_torchvision</li><li>pt_dla_dla102x_visual_bb_torchvision</li><li>pt_dla_dla102_visual_bb_torchvision</li><li>pt_dla_dla169_visual_bb_torchvision</li><li>pt_dla_dla102x2_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 96, 320), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</li><li>pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 160, 160), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 240, 320), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla60x_c_visual_bb_torchvision</li><li>pt_dla_dla46x_c_visual_bb_torchvision</li><li>pt_dla_dla46_c_visual_bb_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 28), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_dla_dla34_visual_bb_torchvision</li><li>pt_dla_dla34_in1k_img_cls_timm</li><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 14), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 147, 147), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 71, 71), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 35, 35), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_v4_img_cls_osmr</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_inception_v4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 17, 17), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 160, 512), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 27, 27), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 13, 13), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 55, 55), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 13, 13), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 54, 54), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 27, 27), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 28, 28), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 14, 14), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_densenet_densenet161_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 1<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 1<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fpn_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8, 8), dtype=float32)</td>
			<td>kernel_size : 1<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 1<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 14, 14), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 1<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 1<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 832, 14, 14), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 832, 7, 7), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 1<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mnist_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 24, 24), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 150, 150), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 32), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 147, 147), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 74, 74), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 37, 37), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 19, 19), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 10, 10), dtype=float32)</td>
			<td>kernel_size : 5<br>stride : 1<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
