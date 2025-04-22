<h1>Comprehensive Report on Reshape Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of reshape operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Reshape Operation Details</th>
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
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1152, 384]</td>
			<td rowspan="1">6</td>
			<td>6</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>shape : (384, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="3">2</td>
			<td rowspan="3">[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
			<td rowspan="3">3</td>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596, 4096), dtype=float32)</td>
			<td>shape : (2441216,)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 4096), dtype=float32)</td>
			<td>shape : (2359296,)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2441216,), dtype=float32)</td>
			<td>shape : (1, 596, 4096)</td>
		</tr>
		<tr>
			<td rowspan="2">3</td>
			<td rowspan="2">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1024, 1024, 3), dtype=float32)</td>
			<td>shape : (1024, 1024, 3, 1)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(1280, 1280, 3), dtype=float32)</td>
			<td>shape : (1280, 1280, 3, 1)</td>
		</tr>
		<tr>
			<td rowspan="1">4</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2304, 768]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 1, 3, 3), dtype=float32)</td>
			<td>shape : (768, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="1">5</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1296, 432]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(432, 1, 3, 3), dtype=float32)</td>
			<td>shape : (432, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="1">6</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2160, 720]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(720, 1, 3, 3), dtype=float32)</td>
			<td>shape : (720, 1, 3, 3)</td>
		</tr>
	</tbody>
</table>
