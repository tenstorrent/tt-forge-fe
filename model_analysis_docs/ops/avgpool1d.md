<h1>Comprehensive Report on AvgPool1d Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of avgpool1d operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Avgpool1D Operation Details</th>
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
			<td rowspan="1">1</td>
			<td rowspan="1">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 49), dtype=float32)</td>
			<td>kernel_size : [49]<br>stride : [49]<br>padding : [0, 0]<br>ceil_mode : False<br>count_include_pad : True</td>
		</tr>
	</tbody>
</table>
