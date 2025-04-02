<h1>Comprehensive Report on Layernorm Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of layernorm operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Layernorm Operation Details</th>
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
			<td rowspan="106">1</td>
			<td rowspan="106">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="106">257</td>
			<td>18</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>16</td>
			<td><ul><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_mlp_mixer_base_img_cls_github</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>6</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-06</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2560,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2560,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2560,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2560,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2560,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2560,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-06</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 7, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 577, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 204, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 4544), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4544,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4544,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 4096), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 2560), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2560,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2560,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(256, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3025, 322), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(322,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(322,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 50176, 261), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(261,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(261,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 0.0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(512,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 2048), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(2048,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 1024), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3136, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 384), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 1536), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1536,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 768), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768,), dtype=float32)</td>
			<td>dim : -1<br>epsilon : 1e-05</td>
		</tr>
	</tbody>
</table>
