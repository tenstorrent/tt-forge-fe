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
			<td rowspan="3">1</td>
			<td rowspan="3">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="3">9</td>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 35, 35), dtype=float32)</td>
			<td>kernel_size : [3, 3]<br>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>ceil_mode : False<br>count_include_pad : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 17, 17), dtype=float32)</td>
			<td>kernel_size : [3, 3]<br>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>ceil_mode : False<br>count_include_pad : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_inception_inception_v4_img_cls_timm</li><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536, 8, 8), dtype=float32)</td>
			<td>kernel_size : [3, 3]<br>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>ceil_mode : False<br>count_include_pad : False<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
