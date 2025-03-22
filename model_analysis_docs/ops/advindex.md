<h1>Comprehensive Report on AdvIndex Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of advindex operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Advindex Operation Details</th>
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
			<td rowspan="26">1</td>
			<td rowspan="26">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="26">46</td>
			<td>12</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_970, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_nbeats_trend_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024, 72), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(448, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.1.0.attn.relative_position_index, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.3.0.attn.relative_position_index, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.5.0.attn.relative_position_index, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.7.0.attn.relative_position_index, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(448, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(448, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(448, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(448, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2359296,), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2441216,), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(7, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_490, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 2), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(732, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(732, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 4), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.1.0.attn.relative_position_index, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 8), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.3.0.attn.relative_position_index, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 16), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.5.0.attn.relative_position_index, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.7.0.attn.relative_position_index, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2401,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2401,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2401,), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(169, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2401,), dtype=int64)</td>
			<td></td>
		</tr>
	</tbody>
</table>
