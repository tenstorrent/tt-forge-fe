<h1>Comprehensive Report on AvgPool3d Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of avgpool3d operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Avgpool3D Operation Details</th>
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
			<td rowspan="2">1</td>
			<td rowspan="2">[MLIR][mlir::AffineMap collapsedLinearAffineMap] python: /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/lib/Dialect/TT/IR/TTOpsTypes.cpp:445: mlir::AffineMap collapsedLinearAffineMap(::mlir::MLIRContext *, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<int64_t>, ::llvm::ArrayRef<std::pair<std::int64_t, std::int64_t>>): Assertion `found && "Dim does not participate in AffineMap RHS"' failed.</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 100, 54, 54), dtype=float32)</td>
			<td>kernel_size : [5, 1, 1]<br>stride : [1, 1, 1]<br>padding : [0, 0, 0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1, 260, 27, 27), dtype=float32)</td>
			<td>kernel_size : [5, 1, 1]<br>stride : [1, 1, 1]<br>padding : [0, 0, 0, 0, 0, 0]<br>ceil_mode : False<br>count_include_pad : True<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
