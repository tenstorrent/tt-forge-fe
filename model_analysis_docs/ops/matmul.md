<h1>Comprehensive Report on Matmul Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of matmul operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Matmul Operation Details</th>
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
			<td rowspan="30">1</td>
			<td rowspan="30">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="30">60</td>
			<td>8</td>
			<td><ul><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg19_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 25088), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(25088, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16384, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 14336), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14336, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(4096, 2), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 14336), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(14336, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li><li>pt_alexnet_base_img_cls_osmr</li><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9216), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(9216, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li><li>DeepSeekWrapper_decoder</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 1), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10240), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(10240, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10240), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(10240, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 10240), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(10240, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 151936), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 8960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8960, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 35, 18944), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18944, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 8192), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8192, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 23040), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(23040, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 10, 9216), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(9216, 3072), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 18176), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18176, 4544), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 16384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16384, 4096), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 16384), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(16384, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 10240), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(10240, 2560), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 8960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8960, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 8960), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(8960, 1536), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 18944), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18944, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 39, 11008), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(11008, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 29, 18944), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(18944, 3584), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 50176), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 50176), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 50176, 512), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="2">2</td>
			<td rowspan="2">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1], got [1, 12]</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.forecast_sin_template, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.forecast_cos_template, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="2">3</td>
			<td rowspan="2">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [72, 1], got [1, 12]</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.backcast_sin_template, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.blocks.0.basis_function.backcast_cos_template, dtype=float32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
