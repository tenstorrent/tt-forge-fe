<h1>Compiler Component Failure Analysis by Model Impacts</h1>
<p>The table highlights the failures encountered in different compiler components, the number of models impacted by each failure, and the specific models affected.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th>Compiler Component</th>
			<th>Failure</th>
			<th>Number of Impacted Models</th>
			<th>Impacted Models</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td rowspan="3">Forge-Fe</td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td>8</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 0 - data type mismatch: expected UInt8, got Float32</td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td rowspan="3">Metalium</td>
			<td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td>6</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn conv2d] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_sharded_program_factory.cpp act_block_w_datums == round_up(conv_act_size_c * filter_w, TILE_WIDTH)</td>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
		</tr>
	</tbody>
</table>
<h1>Compiler-Specific Model Statistics</h1>
<p>The table summarizes model performance across three compiler components: Forge-Fe, MLIR, and Metalium. It highlights the count of models that passed for each component, along with their respective pass rates, average pass rates and the top 10 models with the lowest pass rates.</p>
<ul><li><b>Models Passed: </b>The count of models that achieved a 100% success rate for a specific compiler component.</li><li><b>Pass Rate (%): </b>The percentage of models that successfully passed a specific compiler component, calculated as (Models Passed / Total Number of Models) × 100</li><li><b>Average Pass Rate (%): </b>The mean pass rate for a compiler component, determined by dividing the sum of pass rates of individual models by the total number of models.</li><li><b>Top-10 Blocked Models (pass rate in %): </b>A list of the 10 models with the lowest pass rates for a specific compiler component, including their exact pass rate percentages.</li></ul>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Total no of models : 9</th>
		</tr>
		<tr style="text-align: center;">
			<th>Compiler Component</th>
			<th>Models Passed</th>
			<th>Pass Rate (%)</th>
			<th>Average Pass Rate (%)</th>
			<th>Top-10 Blocked Models (pass rate in %)</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>Forge-Fe</td>
			<td>1</td>
			<td>11 %</td>
			<td>96 %</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf (89 %)</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf (90 %)</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf (90 %)</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf (98 %)</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf (98 %)</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf (99 %)</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf (99 %)</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf (99 %)</li><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf (100 %)</li></ul></td>
		</tr>
		<tr>
			<td>MLIR</td>
			<td>1</td>
			<td>11 %</td>
			<td>96 %</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf (89 %)</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf (90 %)</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf (90 %)</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf (98 %)</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf (98 %)</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf (99 %)</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf (99 %)</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf (99 %)</li><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf (100 %)</li></ul></td>
		</tr>
		<tr>
			<td>Metalium</td>
			<td>1</td>
			<td>11 %</td>
			<td>93 %</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf (82 %)</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf (83 %)</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf (83 %)</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf (97 %)</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf (97 %)</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf (97 %)</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf (97 %)</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf (98 %)</li><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf (100 %)</li></ul></td>
		</tr>
	</tbody>
</table>
<h1>Ops-Specific Failure Statistics</h1>
<p>This table provides detailed insights into operation specific statistics, highlighting the number of failed models for each operation and the associated models that encountered issues. Click on an Operation name to view its detailed analysis</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th>ID</th>
			<th>Operation Name</th>
			<th>Number of failed models</th>
			<th>Failed Models</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>1</td>
			<td><a href="../ops/reshape.md">Reshape</a></td>
			<td>8</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
		</tr>
		<tr>
			<td>2</td>
			<td><a href="../ops/conv2d.md">Conv2d</a></td>
			<td>5</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
		</tr>
		<tr>
			<td>3</td>
			<td><a href="../ops/add.md">Add</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>4</td>
			<td><a href="../ops/cast.md">Cast</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>5</td>
			<td><a href="../ops/embedding.md">Embedding</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>6</td>
			<td><a href="../ops/greater.md">Greater</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>7</td>
			<td><a href="../ops/index.md">Index</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>8</td>
			<td><a href="../ops/layernorm.md">Layernorm</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>9</td>
			<td><a href="../ops/repeatinterleave.md">RepeatInterleave</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>10</td>
			<td><a href="../ops/unsqueeze.md">Unsqueeze</a></td>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
		</tr>
		<tr>
			<td>11</td>
			<td><a href="../ops/matmul.md">Matmul</a></td>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
		</tr>
	</tbody>
</table>
