<h1>Comprehensive Report on Index Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of index operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Index Operation Details</th>
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
			<td rowspan="210">1</td>
			<td rowspan="210">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="210">290</td>
			<td>8</td>
			<td><ul><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2304, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2304, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 768<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2304, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 1536<br>stop : 2304<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 768<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_gpt2_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 1536<br>stop : 2304<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2,), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2,), dtype=float32)</td>
			<td>dim : -1<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 7<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_nanogpt_financialsupport_nanogpt_text_gen_hf</li><li>pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 7, 1024), dtype=uint1)</td>
			<td>dim : -1<br>start : 0<br>stop : 7<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 16<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 16<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 64<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_hrnet_hrnet_w18_small_pose_estimation_timm</li><li>pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 96<br>stop : 112<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 35, 35), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 96<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 35, 35), dtype=float32)</td>
			<td>dim : -3<br>start : 96<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 224, 35, 35), dtype=float32)</td>
			<td>dim : -3<br>start : 160<br>stop : 224<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 17, 17), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 17, 17), dtype=float32)</td>
			<td>dim : -3<br>start : 384<br>stop : 576<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 17, 17), dtype=float32)</td>
			<td>dim : -3<br>start : 576<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 8, 8), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 8, 8), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 640<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_inception_inception_v4_tf_in1k_img_cls_timm</li><li>pt_inception_v4_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 8, 8), dtype=float32)</td>
			<td>dim : -3<br>start : 640<br>stop : 1024<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 53<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 53<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 53<br>stop : 56<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 53<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 56, 96), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 56, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 56<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 25<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 25<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 25<br>stop : 28<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 25<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 28, 28, 192), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 28, 192), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 28<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 3<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 3<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 11<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 11<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 11<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 11<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 14, 384), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li><li>pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 7, 14, 384), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 14<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1), dtype=int64)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1), dtype=int64)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1), dtype=int64)</td>
			<td>dim : -2<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2, 4, 1), dtype=int64)</td>
			<td>dim : -2<br>start : 3<br>stop : 4<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_stereo_facebook_musicgen_small_music_generation_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2048, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Constant, name=clip_model.text_model.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 7<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 64<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deepseek_deepseek_math_7b_instruct_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 39, 128), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 201, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 9<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=bert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 14<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 9<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=bert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 6<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 9, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 6, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 3, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 3, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bloom_bigscience_bloom_1b1_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 16, 3, 96), dtype=float32)</td>
			<td>dim : -2<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 768<br>stop : 1024<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1536<br>stop : 1792<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 2304<br>stop : 2560<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 512<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1280<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 2048<br>stop : 2304<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 2816<br>stop : 3072<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 256<br>stop : 512<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1024<br>stop : 1280<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1792<br>stop : 2048<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(3072, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 2560<br>stop : 2816<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 1<br>stop : 32<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_codegen_salesforce_codegen_350m_mono_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 2</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 10, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 3, 64), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 3, 64), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 334, 64, 3, 64), dtype=float32)</td>
			<td>dim : -2<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 16<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_fuyu_adept_fuyu_8b_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 334, 32), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 16<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gemma_google_gemma_2_2b_it_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 4, 207, 256), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 1024), dtype=uint1)</td>
			<td>dim : -1<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 256, 2048), dtype=uint1)</td>
			<td>dim : -1<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.1.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 6, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 35, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 35, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 14, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 32<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 2, 29, 64), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=roberta.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.embed_positions.weights, dtype=float32)</td>
			<td>dim : -2<br>start : 2<br>stop : 258<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 1024, 72), dtype=float32)</td>
			<td>dim : -1<br>start : -1<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 48), dtype=float32)</td>
			<td>dim : -1<br>start : 12<br>stop : 24<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 48), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 12<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 48), dtype=float32)</td>
			<td>dim : -1<br>start : 36<br>stop : 48<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_nbeats_seasionality_basis_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1024, 48), dtype=float32)</td>
			<td>dim : -1<br>start : 24<br>stop : 36<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(732, 12), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 729<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(732, 12), dtype=float32)</td>
			<td>dim : -2<br>start : 729<br>stop : 732<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 768), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 197<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(732, 16), dtype=float32)</td>
			<td>dim : -2<br>start : 0<br>stop : 729<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(732, 16), dtype=float32)</td>
			<td>dim : -2<br>start : 729<br>stop : 732<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 197, 1024), dtype=float32)</td>
			<td>dim : -2<br>start : 1<br>stop : 197<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)</td>
			<td>dim : -3<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)</td>
			<td>dim : -3<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 64<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 64<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 176, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 160<br>stop : 176<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 288<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 304, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 192<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 304, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 192<br>stop : 288<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 304, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 288<br>stop : 304<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 296, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 160<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 296, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 160<br>stop : 272<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 296, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 272<br>stop : 296<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 128<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 280, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 280<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 112<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 112<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 288<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 416<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 14, 14), dtype=float32)</td>
			<td>dim : -3<br>start : 416<br>stop : 448<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 416<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 448, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 416<br>stop : 448<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 624, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 624, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 384<br>stop : 576<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_googlenet_base_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 624, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 576<br>stop : 624<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 18<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 18<br>stop : 54<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 54<br>stop : 126<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 72<br>stop : 108<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 126, 7, 7), dtype=float32)</td>
			<td>dim : -3<br>start : 108<br>stop : 126<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 36<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 36<br>stop : 54<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 54<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 18<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 18<br>stop : 36<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 28, 28), dtype=float32)</td>
			<td>dim : -3<br>start : 36<br>stop : 72<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 256<br>stop : 512<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 512<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 768<br>stop : 1024<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 1024<br>stop : 1280<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 1280<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1792, 56, 56), dtype=float32)</td>
			<td>dim : -3<br>start : 1536<br>stop : 1792<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 64, 3, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 16, 6, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 4, 12, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_swin_swin_t_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(3, 1, 24, 49, 32), dtype=float32)</td>
			<td>dim : -5<br>start : 2<br>stop : 3<br>stride : 1</td>
		</tr>
	</tbody>
</table>
