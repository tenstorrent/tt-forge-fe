<h1>Comprehensive Report on Gelu Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of gelu operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Gelu Operation Details</th>
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
			<td rowspan="106">209</td>
			<td>17</td>
			<td><ul><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 256), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 512), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1280), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 2048), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 384), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</li><li>pt_mlp_mixer_mixer_b16_224_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 128), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_albert_large_v1_mlm_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_multi_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li><li>pt_codegen_salesforce_codegen_350m_nl_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_clm_hf</li><li>pt_phi2_microsoft_phi_2_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 10240), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1024), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</li><li>pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 512), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 3072), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_large_v2_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 4096), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 8192), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 8192), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16384), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 16384), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls_hf</li><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 3072), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 3072), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_seq_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 11, 10240), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_phi2_microsoft_phi_2_token_cls_hf</li><li>pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 10240), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mlp_mixer_mixer_l16_224_img_cls_timm</li><li>pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16384, 128), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4096, 256), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 640), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1024), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 384), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 768), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 1536), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_swin_s_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_large_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 8192), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_medium_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 1, 6144), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 3000), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 1500), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 1536), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1536), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 3000), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1500), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 5120), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 5120), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 3000), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1500), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 3000), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1500), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 3000), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1500), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1500, 2048), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 2048), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 2048), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 5120), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 204, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1536), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 3072), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 3072), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 18176), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_9b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 14336), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2b_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 16384), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 207, 9216), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 16384), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_1_1_7b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 107, 24576), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 8192), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 3072), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 10240), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8192), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 8192), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1280), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_perceiverio_deepmind_language_perceiver_mlm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2048, 768), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3072, 128), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_base_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 2048), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 2816), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_large_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 2816), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_t5_google_flan_t5_small_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 61, 1024), dtype=float32)</td>
			<td>approximate : "tanh"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1536), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 19200, 256), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4800, 512), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1200, 1280), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 300, 2048), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mlp_mixer_base_img_cls_github</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 256), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 512), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 1024), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 2048), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_b_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 7, 4096), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3136, 384), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 784, 768), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 196, 1536), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 49, 3072), dtype=float32)</td>
			<td>approximate : "none"</td>
		</tr>
	</tbody>
</table>
