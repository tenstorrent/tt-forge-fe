<h1>Comprehensive Report on Resize2d Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of resize2d operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Resize2D Operation Details</th>
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
			<td rowspan="13">1</td>
			<td rowspan="13">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="13">25</td>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 16, 16), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 32, 32), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 64, 64), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 128, 128), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 16, 16), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 32, 32), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64, 64), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 128, 128), dtype=float32)</td>
			<td>sizes : [128, 128]<br>method : "linear"<br>align_corners : False<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)</td>
			<td>sizes : [28, 28]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)</td>
			<td>sizes : [56, 56]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)</td>
			<td>sizes : [112, 112]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)</td>
			<td>sizes : [224, 224]<br>method : "linear"<br>align_corners : True<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
