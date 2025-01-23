<h1>Compiler Component Failure Analysis by Model Impacts</h1>
<p>The table highlights the failures encountered in different compiler components, the number of models impacted by each failure, and the specific models affected.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th>Compiler Component</th>
			<th>Failure</th>
			<th>Number of Impacted Models</th>
			<th>Impacted Models</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td rowspan="6">Forge-Fe</td>
			<td>[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.float32, compiled_model.dtype=torch.bfloat16</td>
			<td>9</td>
			<td><ul><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_albert_base_v1_mlm</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: repeat_interleave</td>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_albert_base_v1_mlm</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype mismatch] E       RuntimeError: Tensor 1 - data type mismatch: expected UInt32, got Float32</td>
			<td>3</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_albert_base_v1_mlm</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][lower_to_mlir] RuntimeError Found Unsupported operations while lowering from TTForge to TTIR in forward graph Unsupported Ops: cumsum</td>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][Runtime Datatype Unsupported] RuntimeError Unhandled dtype Bool</td>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td>[FORGE][mlir generation failure] RuntimeError Generated MLIR module failed verification</td>
			<td>1</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li></ul></td>
		</tr>
		<tr>
			<td rowspan="1">MLIR</td>
			<td>[MLIR][MLIR runtime ttnn ] tt::exception tt-mlir/runtime/lib/ttnn/runtime.cpp Unsupported data type</td>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td rowspan="3">Metalium</td>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError BinaryOpType cannot be mapped to BcastOpMath</td>
			<td>4</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td>3</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast</td>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_opt_facebook_opt_125m_clm</li></ul></td>
		</tr>
	</tbody>
</table>
<h1>Compiler-Specific Model Statistics</h1>
<p>The table summarizes model performance across three compiler components: Forge-Fe, MLIR, and Metalium. It highlights the count of models that passed for each component, along with their respective pass rates, average pass rates and the top 10 models with the lowest pass rates.</p>
<ul><li><b>Models Passed: </b>The count of models that achieved a 100% success rate for a specific compiler component.</li><li><b>Pass Rate (%): </b>The percentage of models that successfully passed a specific compiler component, calculated as (Models Passed / Total Number of Models) × 100</li><li><b>Average Pass Rate (%): </b>The mean pass rate for a compiler component, determined by dividing the sum of pass rates of individual models by the total number of models.</li><li><b>Top-10 Blocked Models (pass rate in %): </b>A list of the 10 models with the lowest pass rates for a specific compiler component, including their exact pass rate percentages.</li></ul>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Total no of models : 10</th>
		</tr>
		<tr style="text-align: center;">
			<th>Compiler Component</th>
			<th>Models Passed</th>
			<th>Pass Rate (%)</th>
			<th>Average Pass Rate (%)</th>
			<th>Top-10 Blocked Models (pass rate in %)</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>Forge-Fe</td>
			<td>1</td>
			<td>10 %</td>
			<td>92 %</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm (89 %)</li><li>pt_albert_base_v1_mlm (90 %)</li><li>pt_gpt2_gpt2_text_gen (90 %)</li><li>pt_opt_facebook_opt_125m_clm (90 %)</li><li>pt_bart_facebook_bart_large_mnli_seq_cls (91 %)</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder (91 %)</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm (91 %)</li><li>pt_xglm_facebook_xglm_564m_clm (93 %)</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm (95 %)</li><li>pt_codegen_salesforce_codegen_350m_mono_clm (100 %)</li></ul></td>
		</tr>
		<tr>
			<td>MLIR</td>
			<td>1</td>
			<td>10 %</td>
			<td>92 %</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen (88 %)</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm (89 %)</li><li>pt_roberta_xlm_roberta_base_mlm (89 %)</li><li>pt_albert_base_v1_mlm (90 %)</li><li>pt_opt_facebook_opt_125m_clm (90 %)</li><li>pt_bart_facebook_bart_large_mnli_seq_cls (91 %)</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder (91 %)</li><li>pt_xglm_facebook_xglm_564m_clm (93 %)</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm (95 %)</li><li>pt_codegen_salesforce_codegen_350m_mono_clm (100 %)</li></ul></td>
		</tr>
		<tr>
			<td>Metalium</td>
			<td>1</td>
			<td>10 %</td>
			<td>90 %</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm (86 %)</li><li>pt_gpt2_gpt2_text_gen (87 %)</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm (87 %)</li><li>pt_opt_facebook_opt_125m_clm (87 %)</li><li>pt_bart_facebook_bart_large_mnli_seq_cls (89 %)</li><li>pt_xglm_facebook_xglm_564m_clm (89 %)</li><li>pt_albert_base_v1_mlm (90 %)</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder (91 %)</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm (95 %)</li><li>pt_codegen_salesforce_codegen_350m_mono_clm (100 %)</li></ul></td>
		</tr>
	</tbody>
</table>
<h1>Ops-Specific Failure Statistics</h1>
<p>This table provides detailed insights into operation specific statistics, highlighting the number of failed models for each operation and the associated models that encountered issues. Click on an Operation name to view its detailed analysis</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th>ID</th>
			<th>Operation Name</th>
			<th>Number of failed models</th>
			<th>Failed Models</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>1</td>
			<td><a href="../ops/embedding.md">Embedding</a></td>
			<td>9</td>
			<td><ul><li>pt_albert_base_v1_mlm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_xglm_facebook_xglm_564m_clm</li></ul></td>
		</tr>
		<tr>
			<td>2</td>
			<td><a href="../ops/repeatinterleave.md">RepeatInterleave</a></td>
			<td>6</td>
			<td><ul><li>pt_albert_base_v1_mlm</li><li>pt_bart_facebook_bart_large_mnli_seq_cls</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_qwen1_5_qwen_qwen1_5_0_5b_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li><li>pt_xglm_facebook_xglm_564m_clm</li></ul></td>
		</tr>
		<tr>
			<td>3</td>
			<td><a href="../ops/index.md">Index</a></td>
			<td>5</td>
			<td><ul><li>pt_albert_base_v1_mlm</li><li>pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</li><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
		<tr>
			<td>4</td>
			<td><a href="../ops/greater.md">Greater</a></td>
			<td>3</td>
			<td><ul><li>pt_bart_facebook_bart_large_mnli_seq_cls</li><li>pt_opt_facebook_opt_125m_clm</li><li>pt_xglm_facebook_xglm_564m_clm</li></ul></td>
		</tr>
		<tr>
			<td>5</td>
			<td><a href="../ops/cast.md">Cast</a></td>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td>6</td>
			<td><a href="../ops/cumsum.md">CumSum</a></td>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
		<tr>
			<td>7</td>
			<td><a href="../ops/max.md">Max</a></td>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li><li>pt_xglm_facebook_xglm_564m_clm</li></ul></td>
		</tr>
		<tr>
			<td>8</td>
			<td><a href="../ops/where.md">Where</a></td>
			<td>2</td>
			<td><ul><li>pt_gpt2_gpt2_text_gen</li><li>pt_gptneo_eleutherai_gpt_neo_125m_clm</li></ul></td>
		</tr>
		<tr>
			<td>9</td>
			<td><a href="../ops/multiply.md">Multiply</a></td>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
		<tr>
			<td>10</td>
			<td><a href="../ops/notequal.md">NotEqual</a></td>
			<td>1</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm</li></ul></td>
		</tr>
	</tbody>
</table>
