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
			<td rowspan="10">1</td>
			<td rowspan="10">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td rowspan="10">42</td>
			<td>11</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1), dtype=int64)</td>
			<td>shape : (1, 1)</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)</td>
			<td>shape : (1, 256)</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)</td>
			<td>shape : (1, 32)</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_t5_small_text_gen_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61), dtype=int64)</td>
			<td>shape : (1, 61)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(8, 1), dtype=int64)</td>
			<td>shape : (2, 4, 1)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1), dtype=int64)</td>
			<td>shape : (2, 1)</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13), dtype=int64)</td>
			<td>shape : (2, 13)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7), dtype=int64)</td>
			<td>shape : (2, 7)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7), dtype=int64)</td>
			<td>shape : (1, 7)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=swin.encoder.layers.0.blocks.0.attention.self.relative_position_index, dtype=int64)</td>
			<td>shape : (2401,)</td>
		</tr>
		<tr>
			<td rowspan="2">2</td>
			<td rowspan="2">[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 0 - data type mismatch: expected UInt8, got Float32</td>
			<td rowspan="2">6</td>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=uint1)</td>
			<td>shape : (1, 1, 1, 128)</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384), dtype=uint1)</td>
			<td>shape : (1, 1, 1, 384)</td>
		</tr>
		<tr>
			<td rowspan="1">3</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1152, 384]</td>
			<td rowspan="1">5</td>
			<td>5</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li><li>pt_mobilenetv2_mobilenetv2_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Parameter, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>shape : (384, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="2">4</td>
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
			<td rowspan="1">5</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2304, 768]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(768, 1, 3, 3), dtype=float32)</td>
			<td>shape : (768, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="1">6</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 864, 288]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(288, 1, 3, 3), dtype=float32)</td>
			<td>shape : (288, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="1">7</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 1296, 432]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(432, 1, 3, 3), dtype=float32)</td>
			<td>shape : (432, 1, 3, 3)</td>
		</tr>
		<tr>
			<td rowspan="1">8</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 0 - stride mismatch: expected [9, 9, 3, 1], got [1, 1, 2160, 720]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(720, 1, 3, 3), dtype=float32)</td>
			<td>shape : (720, 1, 3, 3)</td>
		</tr>
	</tbody>
</table>
