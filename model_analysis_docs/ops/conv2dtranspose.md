<h1>Comprehensive Report on Conv2dTranspose Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of conv2dtranspose operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Conv2Dtranspose Operation Details</th>
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
			<td rowspan="17">1</td>
			<td rowspan="17">[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: conv2d_transpose</td>
			<td rowspan="17">17</td>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4, 16, 2, 2), dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 1, 2, 2), dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 1, 4, 4), dtype=float32)</td>
			<td>stride : 2<br>padding : 1<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 1, 4, 4), dtype=float32)</td>
			<td>stride : 2<br>padding : 1<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 1, 4, 4), dtype=float32)</td>
			<td>stride : 2<br>padding : 1<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 16, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512, 256, 2, 2), dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 2, 2), dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 2, 2), dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 2, 2), dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_125293, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_126293, dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_167293, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_168293, dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_89209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_90209, dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_120209, dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6s_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56132, dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6s_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_75132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76132, dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6n_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 14, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_55132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56132, dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6n_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 28, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_75132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76132, dtype=float32)</td>
			<td>stride : 2<br>padding : 0<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
