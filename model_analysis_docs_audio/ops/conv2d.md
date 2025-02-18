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
			<td rowspan="4">1</td>
			<td rowspan="4">[TT_METAL][ttnn conv2d] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp act_block_w_datums == round_up(conv_act_size_c * filter_w, TILE_WIDTH)</td>
			<td rowspan="4">4</td>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="3">2</td>
			<td rowspan="3">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="3">3</td>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
