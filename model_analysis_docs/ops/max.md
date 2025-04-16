<h1>Comprehensive Report on Max Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of max operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Max Operation Details</th>
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
			<td rowspan="7">1</td>
			<td rowspan="7">[TT_METAL][ttnn elementwise binary] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast</td>
			<td rowspan="7">11</td>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_opt_facebook_opt_125m_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm_hf</li><li>pt_xglm_facebook_xglm_1_7b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_100, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 12, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="1">2</td>
			<td rowspan="1">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(2441216,), dtype=int32)<br><div align='center'>X</div>Operand(type=Constant, name=const_480, dtype=int32)</td>
			<td></td>
		</tr>
	</tbody>
</table>
