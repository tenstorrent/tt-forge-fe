<h1>Comprehensive Report on Operation Failures and Affected Models</h1>
<p>This table provides detailed insights into operation specific statistics, highlighting the number of failed models for each operation and the associated models that encountered issues. Click on an Operation name to view its detailed analysis</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="3">Operation Details</th>
			<th colspan="3">Failure Insight and Impacted Models</th>
		</tr>
		<tr style="text-align: center;">
			<th>ID</th>
			<th>Operands</th>
			<th>Arguments</th>
			<th>Failure</th>
			<th>Number of Models Affected</th>
			<th>Affected Models</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>1</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(30000, 128), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>2</td>
			<td><ul><li>pt_albert_base_v2_mlm</li><li>pt_albert_base_v1_mlm</li></ul></td>
		</tr>
		<tr>
			<td>2</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 128), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>2</td>
			<td><ul><li>pt_albert_base_v2_mlm</li><li>pt_albert_base_v1_mlm</li></ul></td>
		</tr>
		<tr>
			<td>3</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 128), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>2</td>
			<td><ul><li>pt_albert_base_v2_mlm</li><li>pt_albert_base_v1_mlm</li></ul></td>
		</tr>
		<tr>
			<td>4</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50265, 1024), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
		</tr>
		<tr>
			<td>5</td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(1026, 1024), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
		</tr>
		<tr>
			<td>6</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(30522, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder</li></ul></td>
		</tr>
		<tr>
			<td>7</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder</li></ul></td>
		</tr>
		<tr>
			<td>8</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>2</td>
			<td><ul><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li><li>pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder</li></ul></td>
		</tr>
		<tr>
			<td>9</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50257, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>2</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_gpt2_gpt2_text_gen</li></ul></td>
		</tr>
		<tr>
			<td>10</td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
		</tr>
		<tr>
			<td>11</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50257, 2560), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li></ul></td>
		</tr>
		<tr>
			<td>12</td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2560), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_2_7b_clm</li></ul></td>
		</tr>
		<tr>
			<td>13</td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td>14</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50257, 2048), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li></ul></td>
		</tr>
		<tr>
			<td>15</td>
			<td>Operand(type=Constant, name=const_00, dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2048, 2048), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_1_3b_clm</li></ul></td>
		</tr>
		<tr>
			<td>16</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50272, 512), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm</li></ul></td>
		</tr>
		<tr>
			<td>17</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2050, 1024), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm</li></ul></td>
		</tr>
		<tr>
			<td>18</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50272, 2048), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm</li></ul></td>
		</tr>
		<tr>
			<td>19</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2050, 2048), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm</li></ul></td>
		</tr>
		<tr>
			<td>20</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(50272, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td>21</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(2050, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td>22</td>
			<td>Operand(type=Activation, shape=(1, 35), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(151936, 896), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</li></ul></td>
		</tr>
		<tr>
			<td>23</td>
			<td>Operand(type=Activation, shape=(1, 29), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(151936, 896), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_qwen_v2_qwen_qwen2_5_0_5b_clm</li></ul></td>
		</tr>
		<tr>
			<td>24</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(250002, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
		<tr>
			<td>25</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
		<tr>
			<td>26</td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(514, 768), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
		<tr>
			<td>27</td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(256008, 1024), dtype=bfloat16)</td>
			<td></td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>1</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li></ul></td>
		</tr>
	<tbody>
</table>
