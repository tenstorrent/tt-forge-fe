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
			<td rowspan="4">1</td>
			<td rowspan="4">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td rowspan="4">14</td>
			<td>5</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1), dtype=int64)</td>
			<td>shape : (1, 1)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1), dtype=int64)</td>
			<td>shape : (2, 4, 1)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1), dtype=int64)</td>
			<td>shape : (2, 1)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13), dtype=int64)</td>
			<td>shape : (2, 13)</td>
		</tr>
		<tr>
			<td rowspan="2">2</td>
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
	</tbody>
</table>
