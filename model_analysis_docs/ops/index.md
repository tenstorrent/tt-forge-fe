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
			<td rowspan="6">1</td>
			<td rowspan="6">[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td rowspan="6">58</td>
			<td>31</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_distilbert_distilbert_base_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li><li>pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</li><li>pt_distilbert_distilbert_base_uncased_mlm_hf</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>21</td>
			<td><ul><li>pt_albert_xxlarge_v1_token_cls_hf</li><li>pt_albert_large_v1_token_cls_hf</li><li>pt_albert_xlarge_v2_mlm_hf</li><li>pt_albert_base_v2_token_cls_hf</li><li>pt_albert_base_v1_token_cls_hf</li><li>pt_albert_xxlarge_v2_token_cls_hf</li><li>pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</li><li>pt_albert_base_v1_mlm_hf</li><li>pt_albert_xlarge_v2_token_cls_hf</li><li>pt_albert_xxlarge_v1_mlm_hf</li><li>pt_albert_xxlarge_v2_mlm_hf</li><li>pt_albert_xlarge_v1_mlm_hf</li><li>pt_albert_xlarge_v1_token_cls_hf</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_albert_large_v1_mlm_hf</li><li>pt_albert_large_v2_mlm_hf</li><li>pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</li><li>pt_albert_base_v2_mlm_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_bert_bert_base_uncased_mlm_hf</li><li>pt_albert_large_v2_token_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=albert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=bert.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li><li>pt_roberta_xlm_roberta_base_mlm_hf</li></ul></td>
			<td>Operand(type=Constant, name=roberta.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 128<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</li></ul></td>
			<td>Operand(type=Constant, name=clip_model.text_model.embeddings.position_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 7<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li></ul></td>
			<td>Operand(type=Constant, name=bert.embeddings.token_type_ids, dtype=int64)</td>
			<td>dim : -1<br>start : 0<br>stop : 384<br>stride : 1</td>
		</tr>
		<tr>
			<td rowspan="5">2</td>
			<td rowspan="5">[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td rowspan="5">17</td>
			<td>7</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2,), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 1<br>stride : 1</td>
		</tr>
		<tr>
			<td>7</td>
			<td><ul><li>pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</li><li>pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</li><li>pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2,), dtype=float32)</td>
			<td>dim : -1<br>start : 1<br>stop : 2<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 0<br>stop : 768<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 768<br>stop : 1536<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Parameter, shape=(2304,), dtype=float32)</td>
			<td>dim : -1<br>start : 1536<br>stop : 2304<br>stride : 1</td>
		</tr>
		<tr>
			<td rowspan="5">3</td>
			<td rowspan="5">[FORGE][Runtime Datatype Unsupported] RuntimeError Unhandled dtype Bool</td>
			<td rowspan="5">13</td>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.1.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</li><li>pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.1.attn.attention.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 32<br>stride : 1</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen_hf</li></ul></td>
			<td>Operand(type=Constant, name=model.transformer.h.0.attn.bias, dtype=uint1)</td>
			<td>dim : -2<br>start : 0<br>stop : 256<br>stride : 1</td>
		</tr>
	</tbody>
</table>
