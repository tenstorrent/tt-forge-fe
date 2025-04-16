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
			<td rowspan="3">1</td>
			<td rowspan="3">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="3">3</td>
			<td>1</td>
			<td><ul><li>pt_autoencoder_conv_img_enc_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 14, 14), dtype=float32)</td>
			<td>kernel_size : 2<br>stride : 2<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>ceil_mode : False<br>channel_last : 0</td>
		</tr>
		<tr>
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
	</tbody>
</table>
