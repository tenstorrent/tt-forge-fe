<h1>Comprehensive Report on Multiply Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of multiply operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Multiply Operation Details</th>
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
			<td rowspan="63">1</td>
			<td rowspan="63">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="63">100</td>
			<td>5</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_78680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>4</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_157358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_58680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 480, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_149680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_169680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_218680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_25208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_42208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_vovnet_ese_vovnet39b_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</li><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_59208, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_roberta_xlm_roberta_base_mlm_hf</li><li>pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128), dtype=int32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 128), dtype=int32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_9358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_42358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_57358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_102358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_117358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_132358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.3.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_22452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.4.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_31452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.5.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_42452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.6.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_53452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.7.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_64452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.8.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_77452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.9.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_90452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.10.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_103452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.11.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_116452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 54, 54), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_10, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 27, 27), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_40, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.1.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_26680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.1.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_35680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.3.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_70680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.5.0.ghost2.primary_conv.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_96680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.5.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_105680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.6.3.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_161680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ghostnet_ghostnet_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.7.0.shortcut.3.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_198680, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.2.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_11414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.3.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_20414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.4.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_29414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.5.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_38414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.6.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_47414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.7.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_56414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.8.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_65414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.9.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_74414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.10.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_83414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.11.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_92414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.12.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_101414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.13.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_110414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.14.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_119414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.15.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_128414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.16.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_137414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.17.conv.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_146414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_basic_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.18.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_155414, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=segmentation_head.conv_pool.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_155430, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Constant, name=segmentation_head.conv_aspp.normalization.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_158430, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.12.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_131452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.13.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_146452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.14.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_161452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.15.block.0.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_176452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.16.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_191452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Constant, name=features.12.1.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_153358, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_195452, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.21.stack.conv1.bn_pw.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4041166, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.21.stack.conv2.bn_pw.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4101166, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_xception_xception71_img_cls_timm</li></ul></td>
			<td>Operand(type=Constant, name=blocks.21.stack.conv3.bn_pw.running_mean, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_4161166, dtype=float32)</td>
			<td></td>
		</tr>
		<tr>
			<td rowspan="2">2</td>
			<td rowspan="2">[FORGE][TT-Metal vs Forge Output Dtype mismatch] E                   TypeError: Dtype mismatch: framework_model.dtype=torch.int64, compiled_model.dtype=torch.int32</td>
			<td rowspan="2">9</td>
			<td>6</td>
			<td><ul><li>pt_opt_facebook_opt_350m_qa_hf</li><li>pt_opt_facebook_opt_1_3b_seq_cls_hf</li><li>pt_opt_facebook_opt_125m_qa_hf</li><li>pt_opt_facebook_opt_350m_seq_cls_hf</li><li>pt_opt_facebook_opt_1_3b_qa_hf</li><li>pt_opt_facebook_opt_125m_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 32), dtype=int64)</td>
			<td></td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_opt_facebook_opt_350m_clm_hf</li><li>pt_opt_facebook_opt_1_3b_clm_hf</li><li>pt_opt_facebook_opt_125m_clm_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256), dtype=int64)<br><div align='center'>X</div>Operand(type=Activation, shape=(1, 256), dtype=int64)</td>
			<td></td>
		</tr>
	</tbody>
</table>
