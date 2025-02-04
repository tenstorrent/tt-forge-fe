<h1>Comprehensive Report on Softmax Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of softmax operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Softmax Operation Details</th>
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
			<td rowspan="3">[TT_METAL][ttnn softmax] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B Inputs must be of bfloat16 or bfloat8_b type</td>
			<td rowspan="3">6</td>
			<td>2</td>
			<td><ul><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 17, 4, 4480), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 17, 4, 1120), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 17, 4, 280), dtype=float32)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
			<td rowspan="1">2</td>
			<td>2</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512, 50176), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
	</tbody>
</table>
