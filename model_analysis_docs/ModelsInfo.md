<h1>List of models and current compiler support rates</h1>
<p><b>Last updated date and time(in GMT) :</b> Tuesday, 25 Mar 2025 03:10:30 PM</p><p><b>Commit Id :</b> <a href="https://github.com/tenstorrent/tt-forge-fe/commit/505eeba356ba2f9949395a5036b8458433c7f726">505eeba356ba2f9949395a5036b8458433c7f726</a></p><p><b>Note:</b> For detailed insights into compiler failures and their effects on models, please refer to the <a href="./stats/compiler_analysis_report.md">compiler_analysis_report.md</a>.</p><table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Model Details</th>
      <th colspan="4" halign="left">Passing rate of unique ops for each component</th>
    </tr>
    <tr>
      <th></th>
      <th>Name</th>
      <th>Variant</th>
      <th>Framework</th>
      <th>Forge-Fe</th>
      <th>MLIR</th>
      <th>Metalium</th>
      <th>N/A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v1_mlm_hf.md">pt_albert_base_v1_mlm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>2</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v1_token_cls_hf.md">pt_albert_base_v1_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v2_mlm_hf.md">pt_albert_base_v2_mlm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v2_token_cls_hf.md">pt_albert_base_v2_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>5</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf.md">pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>6</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_twmkn9_albert_base_v2_squad2_qa_hf.md">pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>7</th>
      <td>alexnet</td>
      <td><a href="./models/alexnet/pt_alexnet_alexnet_img_cls_torchhub.md">pt_alexnet_alexnet_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>8</th>
      <td>autoencoder</td>
      <td><a href="./models/autoencoder/pt_autoencoder_conv_img_enc_github.md">pt_autoencoder_conv_img_enc_github</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>9</th>
      <td>autoencoder</td>
      <td><a href="./models/autoencoder/pt_autoencoder_linear_img_enc_github.md">pt_autoencoder_linear_img_enc_github</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>10</th>
      <td>bart</td>
      <td><a href="./models/bart/pt_bart_facebook_bart_large_mnli_seq_cls_hf.md">pt_bart_facebook_bart_large_mnli_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>11</th>
      <td>beit</td>
      <td><a href="./models/beit/pt_beit_microsoft_beit_base_patch16_224_img_cls_hf.md">pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>12</th>
      <td>beit</td>
      <td><a href="./models/beit/pt_beit_microsoft_beit_large_patch16_224_img_cls_hf.md">pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>13</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_bert_base_uncased_mlm_hf.md">pt_bert_bert_base_uncased_mlm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>14</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf.md">pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>15</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf.md">pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>16</th>
      <td>bloom</td>
      <td><a href="./models/bloom/pt_bloom_bigscience_bloom_1b1_clm_hf.md">pt_bloom_bigscience_bloom_1b1_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>17</th>
      <td>clip</td>
      <td><a href="./models/clip/pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text.md">pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>18</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_mono_clm_hf.md">pt_codegen_salesforce_codegen_350m_mono_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>19</th>
      <td>deepseek_math</td>
      <td><a href="./models/deepseek_math/pt_deepseek_deepseek_math_7b_instruct_qa_hf.md">pt_deepseek_deepseek_math_7b_instruct_qa_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>20</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_base_patch16_224_img_cls_hf.md">pt_deit_facebook_deit_base_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>21</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet121_hf_xray_img_cls_torchvision.md">pt_densenet_densenet121_hf_xray_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>22</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet161_img_cls_torchvision.md">pt_densenet_densenet161_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>23</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet169_img_cls_torchvision.md">pt_densenet_densenet169_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>24</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_uncased_mlm_hf.md">pt_distilbert_distilbert_base_uncased_mlm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>25</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla34_in1k_img_cls_timm.md">pt_dla_dla34_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>26</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla34_visual_bb_torchvision.md">pt_dla_dla34_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>27</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder.md">pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>28</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b0_img_cls_timm.md">pt_efficientnet_efficientnet_b0_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>29</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b0_img_cls_torchvision.md">pt_efficientnet_efficientnet_b0_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>30</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>31</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>32</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>33</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>34</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>35</th>
      <td>falcon</td>
      <td><a href="./models/falcon/pt_falcon3_tiiuae_falcon3_1b_base_clm_hf.md">pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>36</th>
      <td>fpn</td>
      <td><a href="./models/fpn/pt_fpn_base_img_cls_torchvision.md">pt_fpn_base_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>37</th>
      <td>fuyu_8b</td>
      <td><a href="./models/fuyu_8b/pt_fuyu_adept_fuyu_8b_qa_hf.md">pt_fuyu_adept_fuyu_8b_qa_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>38</th>
      <td>gemma_2b</td>
      <td><a href="./models/gemma_2b/pt_gemma_google_gemma_2_2b_it_qa_hf.md">pt_gemma_google_gemma_2_2b_it_qa_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>39</th>
      <td>ghostnet</td>
      <td><a href="./models/ghostnet/pt_ghostnet_ghostnet_100_img_cls_timm.md">pt_ghostnet_ghostnet_100_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>40</th>
      <td>ghostnet</td>
      <td><a href="./models/ghostnet/pt_ghostnet_ghostnet_100_in1k_img_cls_timm.md">pt_ghostnet_ghostnet_100_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>41</th>
      <td>ghostnet</td>
      <td><a href="./models/ghostnet/pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm.md">pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>42</th>
      <td>googlenet</td>
      <td><a href="./models/googlenet/pt_googlenet_base_img_cls_torchvision.md">pt_googlenet_base_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>43</th>
      <td>gpt2</td>
      <td><a href="./models/gpt2/pt_gpt2_gpt2_text_gen_hf.md">pt_gpt2_gpt2_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>44</th>
      <td>gpt2</td>
      <td><a href="./models/gpt2/pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf.md">pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>45</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_125m_clm_hf.md">pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>46</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm.md">pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>47</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_pose_estimation_timm.md">pt_hrnet_hrnet_w18_small_pose_estimation_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>48</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr.md">pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>49</th>
      <td>inception_v4</td>
      <td><a href="./models/inception_v4/pt_inception_inception_v4_tf_in1k_img_cls_timm.md">pt_inception_inception_v4_tf_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>50</th>
      <td>inception_v4</td>
      <td><a href="./models/inception_v4/pt_inception_v4_img_cls_osmr.md">pt_inception_v4_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>51</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_base_img_cls_github.md">pt_mlp_mixer_base_img_cls_github</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>52</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm.md">pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>53</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_b16_224_img_cls_timm.md">pt_mlp_mixer_mixer_b16_224_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>54</th>
      <td>mnist</td>
      <td><a href="./models/mnist/pt_mnist_base_img_cls_github.md">pt_mnist_base_img_cls_github</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>55</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilenet_v1_basic_img_cls_torchvision.md">pt_mobilenet_v1_basic_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>56</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm.md">pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>57</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_basic_img_cls_torchhub.md">pt_mobilenetv2_basic_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>58</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_mobilenet_v2_img_cls_torchvision.md">pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>59</th>
      <td>mobilenet_v3</td>
      <td><a href="./models/mobilenet_v3/pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub.md">pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>60</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet101_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>61</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet152_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>62</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet18_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>63</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet34_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>64</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet50_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>65</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_1024x320_depth_prediction_torchvision.md">pt_monodepth2_mono_1024x320_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>66</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_640x192_depth_prediction_torchvision.md">pt_monodepth2_mono_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>67</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision.md">pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>68</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision.md">pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>69</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision.md">pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>70</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision.md">pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>71</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_stereo_1024x320_depth_prediction_torchvision.md">pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>72</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_stereo_640x192_depth_prediction_torchvision.md">pt_monodepth2_stereo_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>73</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision.md">pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>74</th>
      <td>monodle</td>
      <td><a href="./models/monodle/pt_monodle_base_obj_det_torchvision.md">pt_monodle_base_obj_det_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>75</th>
      <td>nanogpt</td>
      <td><a href="./models/nanogpt/pt_nanogpt_financialsupport_nanogpt_text_gen_hf.md">pt_nanogpt_financialsupport_nanogpt_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>76</th>
      <td>nbeats</td>
      <td><a href="./models/nbeats/pt_nbeats_seasionality_basis_clm_hf.md">pt_nbeats_seasionality_basis_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>77</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_clm_hf.md">pt_opt_facebook_opt_125m_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>78</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_language_perceiver_mlm_hf.md">pt_perceiverio_deepmind_language_perceiver_mlm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>79</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf.md">pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>80</th>
      <td>qwen</td>
      <td><a href="./models/qwen/pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf.md">pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>81</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf.md">pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>82</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>83</th>
      <td>rcnn</td>
      <td><a href="./models/rcnn/pt_rcnn_base_obj_det_torchvision_rect_0.md">pt_rcnn_base_obj_det_torchvision_rect_0</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>84</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_facebook_regnet_y_040_img_cls_hf.md">pt_regnet_facebook_regnet_y_040_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>85</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_400mf_img_cls_torchvision.md">pt_regnet_regnet_y_400mf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>86</th>
      <td>resnet</td>
      <td><a href="./models/resnet/ResNetForImageClassification.md">ResNetForImageClassification</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>87</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_50_img_cls_timm.md">pt_resnet_50_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>88</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet101_img_cls_torchvision.md">pt_resnet_resnet101_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>89</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet152_img_cls_torchvision.md">pt_resnet_resnet152_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>90</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet18_img_cls_torchvision.md">pt_resnet_resnet18_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>91</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet34_img_cls_torchvision.md">pt_resnet_resnet34_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>92</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet50_img_cls_torchvision.md">pt_resnet_resnet50_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>93</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext101_32x8d_img_cls_torchhub.md">pt_resnext_resnext101_32x8d_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>94</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext50_32x4d_img_cls_torchhub.md">pt_resnext_resnext50_32x4d_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>95</th>
      <td>roberta</td>
      <td><a href="./models/roberta/pt_roberta_xlm_roberta_base_mlm_hf.md">pt_roberta_xlm_roberta_base_mlm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>96</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b0_img_cls_hf.md">pt_segformer_nvidia_mit_b0_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>97</th>
      <td>squeezebert</td>
      <td><a href="./models/squeezebert/pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf.md">pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>98</th>
      <td>ssd300_resnet50</td>
      <td><a href="./models/ssd300_resnet50/pt_ssd300_resnet50_base_img_cls_torchhub.md">pt_ssd300_resnet50_base_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>99</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_small_music_generation_hf.md">pt_stereo_facebook_musicgen_small_music_generation_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>100</th>
      <td>swin</td>
      <td><a href="./models/swin/pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf.md">pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>101</th>
      <td>swin</td>
      <td><a href="./models/swin/pt_swin_swin_t_img_cls_torchvision.md">pt_swin_swin_t_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>102</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_google_flan_t5_small_text_gen_hf.md">pt_t5_google_flan_t5_small_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>103</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_base_text_gen_hf.md">pt_t5_t5_base_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>104</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_large_text_gen_hf.md">pt_t5_t5_large_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>105</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_small_text_gen_hf.md">pt_t5_t5_small_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>106</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_carvana_base_img_seg_github.md">pt_unet_carvana_base_img_seg_github</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>107</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_cityscape_img_seg_osmr.md">pt_unet_cityscape_img_seg_osmr</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>108</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg11_img_cls_torchvision.md">pt_vgg_vgg11_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>109</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg11_obj_det_osmr.md">pt_vgg_vgg11_obj_det_osmr</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>110</th>
      <td>vilt</td>
      <td><a href="./models/vilt/pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf.md">pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>111</th>
      <td>vit</td>
      <td><a href="./models/vit/pt_vit_google_vit_base_patch16_224_img_cls_hf.md">pt_vit_google_vit_base_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>112</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub.md">pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>113</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_vovnet27s_obj_det_osmr.md">pt_vovnet_vovnet27s_obj_det_osmr</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>114</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_tiny_speech_recognition_hf.md">pt_whisper_openai_whisper_tiny_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>115</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet50_2_img_cls_torchvision.md">pt_wideresnet_wide_resnet50_2_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>116</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception71_tf_in1k_img_cls_timm.md">pt_xception_xception71_tf_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>117</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception_img_cls_timm.md">pt_xception_xception_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>118</th>
      <td>xglm</td>
      <td><a href="./models/xglm/pt_xglm_facebook_xglm_564m_clm_hf.md">pt_xglm_facebook_xglm_564m_clm_hf</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
    <tr>
      <th>119</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_imgcls_torchhub_320x320.md">pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>0 %</td>
      <td>100 %</td>
    </tr>
  </tbody>
</table>
