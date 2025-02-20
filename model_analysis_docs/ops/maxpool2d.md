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
			<td rowspan="10">1</td>
			<td rowspan="10">[TT_METAL][ttnn.reshape] RuntimeError tt-metal/ttnn/cpp/ttnn/tensor/tensor_utils.cpp new_volume == old_volume Invalid arguments to reshape</td>
			<td rowspan="10">22</td>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 56, 56), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_vovnet_vovnet39_obj_det_osmr</li><li>pt_vovnet_vovnet57_obj_det_osmr</li><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 14, 14), dtype=float32)</td>
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
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
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
			<td>Operand(type=Activation, shape=(1, 192, 56, 56), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 28, 28), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_vovnet27s_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 14, 14), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="2">2</td>
			<td rowspan="2">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 528, 14, 14), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 1<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 728, 37, 37), dtype=float32)</td>
			<td>kernel_size : 3<br>stride : 2<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">3</td>
			<td rowspan="1">[TT_METAL][ttnn shared operation] RuntimeError ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp (*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal.get_alignment(HalMemType::L1) == 0 Shard page size must currently have L1 aligned page size</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 14, 14), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
