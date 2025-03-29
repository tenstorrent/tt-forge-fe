<h1>Comprehensive Report on Subtract Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of subtract operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Subtract Operation Details</th>
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
			<td rowspan="53">1</td>
			<td rowspan="53">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="53">196</td>
			<td>22</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_10, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_60, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_70, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_50, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 192, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_0160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 6, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 12, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 12, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 24, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 24, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 48, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 48, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 96, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 96, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 192, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_2115, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_1115, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1,), dtype=int32)<br><div align='center'>X</div>Operand(type=Constant, name=const_520, dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_4153, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 1, 1, 13), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_00, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_50, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nbeats_trend_basis_clm_hf</li><li>pt_nbeats_generic_basis_clm_hf</li><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 72), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 72), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 320, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_0160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 20, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_67160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 20, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 40, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_73160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 40, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_76160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 80, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_79160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 80, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_82160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 160, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_85160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 160, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_88160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_1024x320_depth_prediction_torchvision</li><li>pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 320, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_91160, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_50, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 7), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_70, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_60, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 32, 32), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_40, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_10, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 6), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Constant, name=const_20, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 1, 7, 7), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2441216,), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_470, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_10, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 204), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_10, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 201), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_10, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 14), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_10, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 9), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)<br><div align='center'>X</div>Operand(type=Constant, name=const_2445, dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_2115, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 384), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_1115, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 12, 384, 384), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_50, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_1940, dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 4), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_20, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 2048), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_80, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 256, 256), dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=const_30, dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 1, 1, 128), dtype=float32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
