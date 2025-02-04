<h1>Comprehensive Report on Transpose Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of transpose operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Transpose Operation Details</th>
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
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [1280, 1], got [1, 1001]</td>
			<td rowspan="1">3</td>
			<td>3</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1001, 1280), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [768, 1], got [1, 1001]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1001, 768), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
		<tr>
			<td rowspan="1">3</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [1024, 1], got [1, 1001]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1001, 1024), dtype=float32)</td>
			<td>dim0 : -2<br>dim1 : -1</td>
		</tr>
	</tbody>
</table>
