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
			<td rowspan="7">1</td>
			<td rowspan="7">[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
			<td rowspan="7">12</td>
			<td>6</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 255, 6400), dtype=float32)</td>
			<td>shape : (1, 3, 85, 6400)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 6144), dtype=float32)</td>
			<td>shape : (2, 1, 6144)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 6144), dtype=float32)</td>
			<td>shape : (2, 6144)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 8192), dtype=float32)</td>
			<td>shape : (2, 1, 8192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 8192), dtype=float32)</td>
			<td>shape : (2, 8192)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 12288), dtype=float32)</td>
			<td>shape : (1, 334, 64, 3, 64)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 255, 25600), dtype=float32)</td>
			<td>shape : (1, 3, 85, 25600)</td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1152, 384]</td>
			<td rowspan="1">5</td>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>shape : (384, 1, 3, 3)</td>
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
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 864, 288]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(288, 1, 3, 3), dtype=float32)</td>
			<td>shape : (288, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="1">6</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1296, 432]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(432, 1, 3, 3), dtype=float32)</td>
			<td>shape : (432, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="1">7</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2160, 720]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(720, 1, 3, 3), dtype=float32)</td>
			<td>shape : (720, 1, 3, 3)</td>
		</tr>
	</tbody>
</table>
