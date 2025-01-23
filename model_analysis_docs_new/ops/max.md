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
			<td>Operand(type=Activation, shape=(1, 16, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast</td>
			<td>2</td>
			<td><ul><li>pt_xglm_facebook_xglm_564m_clm</li><li>pt_opt_facebook_opt_350m_clm</li></ul></td>
		</tr>
		<tr>
			<td>2</td>
			<td>Operand(type=Activation, shape=(1, 32, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast</td>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_1_3b_clm</li></ul></td>
		</tr>
		<tr>
			<td>3</td>
			<td>Operand(type=Activation, shape=(1, 12, 256, 256), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110, dtype=float32)</td>
			<td></td>
			<td>[TT_METAL][ttnn elementwise binary] RuntimeError tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.cpp ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast</td>
			<td>1</td>
			<td><ul><li>pt_opt_facebook_opt_125m_clm</li></ul></td>
		</tr>
	<tbody>
</table>
