<h1>Comprehensive Report on Conv2d Operation Failures and Affected Models</h1>
<p>The table presents a detailed summary of conv2d operation failures, including failure descriptions, the total number of affected models, and the specific models impacted. It also provides insights into the operand configurations and associated arguments for each failure, delivering a comprehensive view of the encountered issues.</p>
<table border="2">
	<thead>
		<tr style="text-align: center;">
			<th colspan="5">Failure Insight and Impacted Models</th>
			<th colspan="2">Conv2D Operation Details</th>
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
			<td rowspan="72">1</td>
			<td rowspan="72">[TT_METAL][ttmetal allocations] RuntimeError Statically allocated circular buffers</td>
			<td rowspan="72">107</td>
			<td>10</td>
			<td><ul><li>pt_vgg_vgg19_obj_det_osmr</li><li>pt_vgg_vgg19_bn_obj_det_torchhub</li><li>pt_vgg_bn_vgg19_obj_det_osmr</li><li>pt_vgg_vgg13_obj_det_osmr</li><li>pt_vgg_19_obj_det_hf</li><li>pt_vgg_vgg19_bn_obj_det_timm</li><li>pt_unet_cityscape_img_seg_osmr</li><li>pt_vgg_vgg16_obj_det_osmr</li><li>pt_vgg_vgg11_obj_det_osmr</li><li>pt_vgg_bn_vgg19b_obj_det_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>9</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b1_img_cls_hf</li><li>pt_segformer_nvidia_mit_b3_img_cls_hf</li><li>pt_segformer_nvidia_mit_b4_img_cls_hf</li><li>pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b5_img_cls_hf</li><li>pt_segformer_nvidia_mit_b2_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 32, 32), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1280<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 480, 480), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.0.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_efficientnet_efficientnet_b0_img_cls_torchvision</li><li>pt_efficientnet_efficientnet_b0_img_cls_timm</li><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</li><li>pt_segformer_nvidia_mit_b0_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 512, 512), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 7, 7), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolo_v6_yolov6s_obj_det_torchhub</li><li>pt_yolo_v6_yolov6n_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 448, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1132, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2132, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolox_yolox_l_obj_det_torchhub</li><li>pt_yolox_yolox_darknet_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 64, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_yolox_yolox_l_obj_det_torchhub</li><li>pt_yolox_yolox_darknet_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_small_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_medium_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1024, 1024, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_base_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 512, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 80, 3, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_large_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1280, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1280, 1280, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_base_img_cls_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 3, 11, 11), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_alexnet_alexnet_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 11, 11), dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 672, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(672, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 672<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_efficientnet_efficientnet_b4_img_cls_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(960, 1, 5, 5), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 960<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_monodle_base_obj_det_torchvision</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(1792, 64, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_rcnn_base_obj_det_torchvision_rect_0</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 227, 227), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=features.0.weight, dtype=float32)</td>
			<td>stride : [4, 4]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(324, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_base_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 64, 64), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 256, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 384, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(128, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_cityscape_img_seg_osmr</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vovnet_ese_vovnet99b_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 512, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.3.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.5.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 240, 240), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.1.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 120, 120), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.3.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 120, 120), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.3.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.1.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.3.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.1.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.3.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.5.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.1.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.2.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.3.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.4.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.5.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.6.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.7.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.8.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.9.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.10.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.6.m.11.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.m.1.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.m.2.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.13.m.3.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.18.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.m.0.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.m.1.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.m.2.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.20.m.3.cv2.conv.weight, dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 448, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1293, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2293, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6l_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 112, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_23293, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_24293, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v6_yolov6m_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 448, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_1209, dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=const_2209, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_darknet_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 128, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_x_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 160, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_x_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(640, 320, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_x_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 320, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_x_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 320, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_x_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 320, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_m_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 48, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="24">2</td>
			<td rowspan="24">[MLIR][TTIR to TTNN Conv2dOpConversionPattern] tt_forge_signal_handler tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp Conv2dOpConversionPattern::matchAndRewrite(ttir::Conv2dOp, OpAdaptor, ConversionPatternRewriter &) adaptor.getPaddingBottom() == adaptor.getPaddingTop() TTNN only supports padding height/width attributes. Thus, padding_top must equal padding_bottom for the op to execute as expected</td>
			<td rowspan="24">28</td>
			<td>3</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>2</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 192, 192), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 384<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 112, 112), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(64, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 64<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 128, 56, 56), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(128, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 128<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(256, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 256<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(512, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 512<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(576, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 576<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 160, 160), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(144, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 144<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(432, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 432<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 3, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(48, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 48<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(96, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 96<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(192, 1, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [0, 1, 0, 1]<br>dilation : 1<br>groups : 192<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="12">3</td>
			<td rowspan="12">[UNKNOWN] The failure does not match any known compiler component exception rules. Please review the failure log to identify the component</td>
			<td rowspan="12">22</td>
			<td>5</td>
			<td><ul><li>pt_retinanet_retinanet_rn101fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn34fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn50fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn152fpn_obj_det_hf</li><li>pt_retinanet_retinanet_rn18fpn_obj_det_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 480, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 3, 7, 7), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [3, 3, 3, 3]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>5</td>
			<td><ul><li>pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li><li>pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 640, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.0.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>3</td>
			<td><ul><li>pt_deit_facebook_deit_base_patch16_224_img_cls_hf</li><li>pt_vit_google_vit_base_patch16_224_img_cls_hf</li><li>pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 3, 16, 16), dtype=float32)</td>
			<td>stride : [16, 16]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 3, 16, 16), dtype=float32)</td>
			<td>stride : [16, 16]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_deit_facebook_deit_small_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 3, 16, 16), dtype=float32)</td>
			<td>stride : [16, 16]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vit_google_vit_large_patch16_224_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 224, 224), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1024, 3, 16, 16), dtype=float32)</td>
			<td>stride : [16, 16]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 1280, 1280), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.0.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [2, 2, 2, 2]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 640, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.1.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Constant, name=model.model.model.1.conv.weight, dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_x_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 80, 320, 320), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 80, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_darknet_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3, 640, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 3, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_yolox_yolox_darknet_obj_det_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 640, 640), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 32, 3, 3), dtype=float32)</td>
			<td>stride : [2, 2]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="4">4</td>
			<td rowspan="4">[TT_METAL][TT-Metal vs Forge Output Data mismatch] ValueError Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model , compiled_model</td>
			<td rowspan="4">4</td>
			<td>1</td>
			<td><ul><li>pt_whisper_openai_whisper_tiny_speech_recognition_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 3000, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(384, 384, 3, 1), dtype=float32)</td>
			<td>stride : [2, 1]<br>padding : [0, 0, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3072, 1, 128), dtype=float32)<br><div align='center'>X</div>Operand(type=Activation, shape=(768, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 4<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_unet_qubvel_img_seg_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 3072, 14, 14), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 3072, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_vgg_vgg19_bn_obj_det_timm</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 512, 7, 7), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(4096, 512, 7, 7), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="3">5</td>
			<td rowspan="3">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 144, 3456, 3456]</td>
			<td rowspan="3">3</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(144, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="3">6</td>
			<td rowspan="3">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 24, 3456, 3456]</td>
			<td rowspan="3">3</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 20, 20), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="2">7</td>
			<td rowspan="2">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [16, 1, 1, 1], got [1, 96, 1536, 1536]</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="2">8</td>
			<td rowspan="2">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 24, 2304, 2304]</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 40, 40), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(24, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="2">9</td>
			<td rowspan="2">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [320, 1, 1, 1], got [1, 256, 81920, 81920]</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 1, 1), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 320, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(256, 320, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="2">10</td>
			<td rowspan="2">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [8, 1, 1, 1], got [1, 48, 384, 384]</td>
			<td rowspan="2">2</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 8, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 8, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">11</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 48, 1152, 1152]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 96, 96), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">12</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 96, 4608, 4608]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">13</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 96, 9216, 9216]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">14</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 192, 18432, 18432]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">15</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 192, 36864, 36864]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">16</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 384, 73728, 73728]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">17</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 384, 147456, 147456]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">18</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 768, 294912, 294912]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">19</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [768, 1, 1, 1], got [1, 768, 589824, 589824]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 768, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(768, 768, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">20</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [24, 1, 1, 1], got [1, 16, 384, 384]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 24, 80, 80), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 24, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">21</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 48, 6912, 6912]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">22</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 288, 13824, 13824]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(288, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">23</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [288, 1, 1, 1], got [1, 48, 13824, 13824]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(48, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">24</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [288, 1, 1, 1], got [1, 72, 20736, 20736]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 288, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 288, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">25</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [72, 1, 1, 1], got [1, 432, 31104, 31104]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 72, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(432, 72, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">26</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [432, 1, 1, 1], got [1, 72, 31104, 31104]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 10, 10), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(72, 432, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">27</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [432, 1, 1, 1], got [1, 120, 51840, 51840]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 432, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 432, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">28</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [120, 1, 1, 1], got [1, 720, 86400, 86400]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 120, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(720, 120, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">29</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [720, 1, 1, 1], got [1, 120, 86400, 86400]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(120, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">30</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [720, 1, 1, 1], got [1, 240, 172800, 172800]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 720, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(240, 720, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">31</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [240, 1, 1, 1], got [1, 1280, 307200, 307200]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 240, 5, 5), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 240, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">32</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 64, 12288, 12288]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">33</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [64, 1, 1, 1], got [1, 384, 24576, 24576]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 64, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(384, 64, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">34</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 64, 24576, 24576]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(64, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">35</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [384, 1, 1, 1], got [1, 96, 36864, 36864]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 384, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 384, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">36</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 576, 55296, 55296]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(576, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">37</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [576, 1, 1, 1], got [1, 96, 55296, 55296]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(96, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">38</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [576, 1, 1, 1], got [1, 160, 92160, 92160]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 576, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 576, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">39</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [160, 1, 1, 1], got [1, 960, 153600, 153600]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 160, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(960, 160, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">40</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [960, 1, 1, 1], got [1, 160, 153600, 153600]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(160, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">41</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [960, 1, 1, 1], got [1, 320, 307200, 307200]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 960, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(320, 960, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">42</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [256, 1, 1, 1], got [1, 21, 5376, 5376]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 256, 28, 28), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(21, 256, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">43</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [16, 1, 1, 1], got [1, 8, 128, 128]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 16, 48, 48), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 16, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">44</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 8, 384, 384]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 24, 24), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(8, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">45</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [48, 1, 1, 1], got [1, 16, 768, 768]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 48, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 48, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">46</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [96, 1, 1, 1], got [1, 16, 1536, 1536]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 96, 12, 12), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 96, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">47</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [144, 1, 1, 1], got [1, 32, 4608, 4608]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 144, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 144, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">48</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [32, 1, 1, 1], got [1, 192, 6144, 6144]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 32, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(192, 32, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">49</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 32, 6144, 6144]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 6, 6), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(32, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">50</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [192, 1, 1, 1], got [1, 56, 10752, 10752]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 192, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 192, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">51</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [56, 1, 1, 1], got [1, 336, 18816, 18816]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 56, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(336, 56, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">52</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [336, 1, 1, 1], got [1, 56, 18816, 18816]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(56, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">53</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [336, 1, 1, 1], got [1, 112, 37632, 37632]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 336, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(112, 336, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">54</td>
			<td rowspan="1">[FORGE][Runtime stride mismatch] E       RuntimeError: Tensor 1 - stride mismatch: expected [112, 1, 1, 1], got [1, 1280, 143360, 143360]</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 112, 3, 3), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(1280, 112, 1, 1), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [0, 0, 0, 0]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
		<tr>
			<td rowspan="1">55</td>
			<td rowspan="1">[TT_METAL][tt-metal ncrisc build] RuntimeError tt-metal/tt_metal/impl/program/program.cpp Failed to generate binaries for reader_conv_activations_padded_with_halo_3x3_weights_v2 ncrisc build failed</td>
			<td rowspan="1">1</td>
			<td>1</td>
			<td><ul><li>pt_ssd300_resnet50_base_img_cls_torchhub</li></ul></td>
			<td>Operand(type=Activation, shape=(1, 1024, 38, 38), dtype=float32)<br><div align='center'>X</div>Operand(type=Parameter, shape=(16, 1024, 3, 3), dtype=float32)</td>
			<td>stride : [1, 1]<br>padding : [1, 1, 1, 1]<br>dilation : 1<br>groups : 1<br>channel_last : 0</td>
		</tr>
	</tbody>
</table>
