<h1>Comprehensive Report on Unsqueeze Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of unsqueeze operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Unsqueeze Operation Details</th>
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
			<td rowspan="13">1</td>
			<td rowspan="13">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td rowspan="13">83</td>
			<td>23</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>23</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 128), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 32), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 13), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 13), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 7), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7), dtype=int64)</td>
			<td>dim : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7), dtype=int64)</td>
			<td>dim : 2</td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 0 - data type mismatch: expected UInt8, got Float32</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 596), dtype=uint1)</td>
			<td>dim : -1</td>
		</tr>
	</tbody>
</table>
