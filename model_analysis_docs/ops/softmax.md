<h1>Comprehensive Report on Softmax Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of softmax operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Softmax Operation Details</th>
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
			<td rowspan="118">1</td>
			<td rowspan="118">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="118">253</td>
			<td>20</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 128, 128), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>12</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 16384, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 4096, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>11</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 1024, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>10</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_clm_hf</li><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</li><li>pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 128, 128), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</li><li>pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</li><li>pt_llama3_huggyllama_llama_7b_seq_cls_hf</li><li>pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 4, 4), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 128, 128), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 256, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 197, 197), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 12, 13, 13), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li><li>pt_t5_t5_base_text_gen_hf</li><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 512, 512), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1, 512), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 3, 49, 49), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 6, 49, 49), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 12, 49, 49), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 49, 49), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 384, 384), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 32), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</li><li>pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10, 10), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 7, 7), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 256, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 32, 32), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 32, 32), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 32, 32), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512, 50176), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 11, 11), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 12, 12), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 29, 29), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 35, 35), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 35, 35), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</li><li>pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 35, 35), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 61, 61), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_t5_base_text_gen_hf</li><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 61), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 61, 61), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li><li>pt_t5_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 61), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 197, 197), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 6, 6), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 1, 13), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 1, 13), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(48, 1, 13), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1500, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1, 1), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1500, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 1, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1500, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 1, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1500, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1500, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 2, 2), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 2, 1500), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 7, 7), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 588, 588), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 39), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 577, 577), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 596, 596), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 204, 204), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 201, 201), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 14, 14), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 9, 9), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 384, 384), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 10), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 71, 6, 6), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 334), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 207, 207), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 7, 7), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 207), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 107, 107), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 107, 107), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 256, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 32, 32), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 20, 32, 32), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 4, 4), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_meta_llama_llama_3_2_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 32, 32), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llama3_huggyllama_llama_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 32), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(32, 256, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(12, 256, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 512, 3025), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 256, 2048), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 2048, 256), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 5, 5), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 13, 13), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 6), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 35), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 39, 39), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 29, 29), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 39, 39), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 39, 39), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 39, 39), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 29), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 13, 13), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 29, 29), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 61, 61), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 1, 61), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 61, 61), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 1, 61), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 197, 197), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 197, 197), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 19200, 300), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 4800, 300), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 5, 1200, 300), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 300, 300), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(64, 4, 49, 49), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(16, 8, 49, 49), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(4, 16, 49, 49), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 49, 49), dtype=float32)</td>
			<td>dim : -1</td>
		</tr>
	</tbody>
</table>
