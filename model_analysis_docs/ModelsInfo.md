<h1>List of models and current compiler support rates</h1>
<p><b>Last updated date and time(in GMT) :</b> Tuesday, 22 Apr 2025 02:00:12 AM</p><p><b>Commit Id :</b> <a href="https://github.com/tenstorrent/tt-forge-fe/commit/b3339ddd9f18ff37b7b919843edf0431212135fe">b3339ddd9f18ff37b7b919843edf0431212135fe</a></p><p><b>Note:</b> For detailed insights into compiler failures and their effects on models, please refer to the <a href="./stats/compiler_analysis_report.md">compiler_analysis_report.md</a>.</p><table border="1" class="dataframe">
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
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>2</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v1_token_cls_hf.md">pt_albert_base_v1_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v2_mlm_hf.md">pt_albert_base_v2_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v2_token_cls_hf.md">pt_albert_base_v2_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>5</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_large_v1_mlm_hf.md">pt_albert_large_v1_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>6</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_large_v1_token_cls_hf.md">pt_albert_large_v1_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>7</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_large_v2_mlm_hf.md">pt_albert_large_v2_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>8</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_large_v2_token_cls_hf.md">pt_albert_large_v2_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>9</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf.md">pt_albert_textattack_albert_base_v2_imdb_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>10</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_twmkn9_albert_base_v2_squad2_qa_hf.md">pt_albert_twmkn9_albert_base_v2_squad2_qa_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>11</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xlarge_v1_mlm_hf.md">pt_albert_xlarge_v1_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>12</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xlarge_v1_token_cls_hf.md">pt_albert_xlarge_v1_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>13</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xlarge_v2_mlm_hf.md">pt_albert_xlarge_v2_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>14</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xlarge_v2_token_cls_hf.md">pt_albert_xlarge_v2_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>15</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xxlarge_v1_mlm_hf.md">pt_albert_xxlarge_v1_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>16</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xxlarge_v1_token_cls_hf.md">pt_albert_xxlarge_v1_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>17</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xxlarge_v2_mlm_hf.md">pt_albert_xxlarge_v2_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>18</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xxlarge_v2_token_cls_hf.md">pt_albert_xxlarge_v2_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>19</th>
      <td>alexnet</td>
      <td><a href="./models/alexnet/pt_alexnet_alexnet_img_cls_torchhub.md">pt_alexnet_alexnet_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>20</th>
      <td>alexnet</td>
      <td><a href="./models/alexnet/pt_alexnet_base_img_cls_osmr.md">pt_alexnet_base_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>87 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>21</th>
      <td>autoencoder</td>
      <td><a href="./models/autoencoder/pt_autoencoder_conv_img_enc_github.md">pt_autoencoder_conv_img_enc_github</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>84 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>22</th>
      <td>autoencoder</td>
      <td><a href="./models/autoencoder/pt_autoencoder_linear_img_enc_github.md">pt_autoencoder_linear_img_enc_github</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>23</th>
      <td>bart</td>
      <td><a href="./models/bart/pt_bart_facebook_bart_large_mnli_seq_cls_hf.md">pt_bart_facebook_bart_large_mnli_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>92 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>24</th>
      <td>beit</td>
      <td><a href="./models/beit/pt_beit_microsoft_beit_base_patch16_224_img_cls_hf.md">pt_beit_microsoft_beit_base_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>25</th>
      <td>beit</td>
      <td><a href="./models/beit/pt_beit_microsoft_beit_large_patch16_224_img_cls_hf.md">pt_beit_microsoft_beit_large_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>26</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_bert_base_uncased_mlm_hf.md">pt_bert_bert_base_uncased_mlm_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>27</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf.md">pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>28</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf.md">pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>29</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf.md">pt_bert_emrecan_bert_base_turkish_cased_mean_nli_stsb_tr_sentence_embed_gen_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>30</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf.md">pt_bert_phiyodr_bert_large_finetuned_squad2_qa_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>31</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf.md">pt_bert_textattack_bert_base_uncased_sst_2_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>32</th>
      <td>bloom</td>
      <td><a href="./models/bloom/pt_bloom_bigscience_bloom_1b1_clm_hf.md">pt_bloom_bigscience_bloom_1b1_clm_hf</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>92 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>33</th>
      <td>clip</td>
      <td><a href="./models/clip/pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text.md">pt_clip_openai_clip_vit_base_patch32_text_gen_hf_text</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>92 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>34</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_mono_clm_hf.md">pt_codegen_salesforce_codegen_350m_mono_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>35</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_multi_clm_hf.md">pt_codegen_salesforce_codegen_350m_multi_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>36</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_nl_clm_hf.md">pt_codegen_salesforce_codegen_350m_nl_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>37</th>
      <td>deepseek_coder</td>
      <td><a href="./models/deepseek_coder/pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf.md">pt_deepseek_deepseek_coder_1_3b_instruct_qa_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>38</th>
      <td>deepseek_math</td>
      <td><a href="./models/deepseek_math/pt_deepseek_deepseek_math_7b_instruct_qa_hf.md">pt_deepseek_deepseek_math_7b_instruct_qa_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>39</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf.md">pt_deit_facebook_deit_base_distilled_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>40</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_base_patch16_224_img_cls_hf.md">pt_deit_facebook_deit_base_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>41</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_small_patch16_224_img_cls_hf.md">pt_deit_facebook_deit_small_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>42</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf.md">pt_deit_facebook_deit_tiny_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>43</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet121_hf_xray_img_cls_torchvision.md">pt_densenet_densenet121_hf_xray_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>44</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet121_img_cls_torchvision.md">pt_densenet_densenet121_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>45</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet161_img_cls_torchvision.md">pt_densenet_densenet161_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>46</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet169_img_cls_torchvision.md">pt_densenet_densenet169_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>47</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet201_img_cls_torchvision.md">pt_densenet_densenet201_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>48</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf.md">pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>92 %</td>
      <td>85 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>49</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf.md">pt_distilbert_distilbert_base_cased_distilled_squad_qa_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>85 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>50</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_cased_mlm_hf.md">pt_distilbert_distilbert_base_cased_mlm_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>85 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>51</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_multilingual_cased_mlm_hf.md">pt_distilbert_distilbert_base_multilingual_cased_mlm_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>85 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>52</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf.md">pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>86 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>53</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_uncased_mlm_hf.md">pt_distilbert_distilbert_base_uncased_mlm_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>85 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>54</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla102_visual_bb_torchvision.md">pt_dla_dla102_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>55</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla102x2_visual_bb_torchvision.md">pt_dla_dla102x2_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>56</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla102x_visual_bb_torchvision.md">pt_dla_dla102x_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>57</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla169_visual_bb_torchvision.md">pt_dla_dla169_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>58</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla34_in1k_img_cls_timm.md">pt_dla_dla34_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>59</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla34_visual_bb_torchvision.md">pt_dla_dla34_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>60</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla46_c_visual_bb_torchvision.md">pt_dla_dla46_c_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>61</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla46x_c_visual_bb_torchvision.md">pt_dla_dla46x_c_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>62</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla60_visual_bb_torchvision.md">pt_dla_dla60_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>63</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla60x_c_visual_bb_torchvision.md">pt_dla_dla60x_c_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>64</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla60x_visual_bb_torchvision.md">pt_dla_dla60x_visual_bb_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>65</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder.md">pt_dpr_facebook_dpr_ctx_encoder_multiset_base_qa_hf_context_encoder</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>66</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder.md">pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_qa_hf_context_encoder</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>67</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder.md">pt_dpr_facebook_dpr_question_encoder_multiset_base_qa_hf_question_encoder</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>68</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder.md">pt_dpr_facebook_dpr_question_encoder_single_nq_base_qa_hf_question_encoder</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>69</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader.md">pt_dpr_facebook_dpr_reader_multiset_base_qa_hf_reader</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>95 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>70</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader.md">pt_dpr_facebook_dpr_reader_single_nq_base_qa_hf_reader</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>95 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>71</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b0_img_cls_timm.md">pt_efficientnet_efficientnet_b0_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>72</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b0_img_cls_torchvision.md">pt_efficientnet_efficientnet_b0_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>73</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b4_img_cls_timm.md">pt_efficientnet_efficientnet_b4_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>74</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b4_img_cls_torchvision.md">pt_efficientnet_efficientnet_b4_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>75</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite0_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>76</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite1_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>77</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite2_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>78</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite3_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>79</th>
      <td>efficientnet_lite</td>
      <td><a href="./models/efficientnet_lite/pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm.md">pt_efficientnet_lite_tf_efficientnet_lite4_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>80</th>
      <td>falcon</td>
      <td><a href="./models/falcon/pt_falcon3_tiiuae_falcon3_1b_base_clm_hf.md">pt_falcon3_tiiuae_falcon3_1b_base_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>81</th>
      <td>falcon</td>
      <td><a href="./models/falcon/pt_falcon3_tiiuae_falcon3_3b_base_clm_hf.md">pt_falcon3_tiiuae_falcon3_3b_base_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>82</th>
      <td>falcon</td>
      <td><a href="./models/falcon/pt_falcon3_tiiuae_falcon3_7b_base_clm_hf.md">pt_falcon3_tiiuae_falcon3_7b_base_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>83</th>
      <td>falcon</td>
      <td><a href="./models/falcon/pt_falcon_tiiuae_falcon_7b_instruct_clm_hf.md">pt_falcon_tiiuae_falcon_7b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>84</th>
      <td>fpn</td>
      <td><a href="./models/fpn/pt_fpn_base_img_cls_torchvision.md">pt_fpn_base_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>85</th>
      <td>fuyu_8b</td>
      <td><a href="./models/fuyu_8b/pt_fuyu_adept_fuyu_8b_qa_hf.md">pt_fuyu_adept_fuyu_8b_qa_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>86</th>
      <td>gemma_2b</td>
      <td><a href="./models/gemma_2b/pt_gemma_google_gemma_2_2b_it_qa_hf.md">pt_gemma_google_gemma_2_2b_it_qa_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>87</th>
      <td>gemma_2b</td>
      <td><a href="./models/gemma_2b/pt_gemma_google_gemma_2_9b_it_qa_hf.md">pt_gemma_google_gemma_2_9b_it_qa_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>88</th>
      <td>gemma_2b</td>
      <td><a href="./models/gemma_2b/pt_gemma_google_gemma_2b_text_gen_hf.md">pt_gemma_google_gemma_2b_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>89</th>
      <td>gemma_v1</td>
      <td><a href="./models/gemma_v1/pt_gemma_google_gemma_1_1_2b_it_qa_hf.md">pt_gemma_google_gemma_1_1_2b_it_qa_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>90</th>
      <td>gemma_v1</td>
      <td><a href="./models/gemma_v1/pt_gemma_google_gemma_1_1_7b_it_qa_hf.md">pt_gemma_google_gemma_1_1_7b_it_qa_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>91</th>
      <td>ghostnet</td>
      <td><a href="./models/ghostnet/pt_ghostnet_ghostnet_100_img_cls_timm.md">pt_ghostnet_ghostnet_100_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>92</th>
      <td>ghostnet</td>
      <td><a href="./models/ghostnet/pt_ghostnet_ghostnet_100_in1k_img_cls_timm.md">pt_ghostnet_ghostnet_100_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>93</th>
      <td>ghostnet</td>
      <td><a href="./models/ghostnet/pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm.md">pt_ghostnet_ghostnetv2_100_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>94</th>
      <td>glpn_kitti</td>
      <td><a href="./models/glpn_kitti/pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf.md">pt_glpn_kitti_vinvino02_glpn_kitti_depth_estimation_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>95</th>
      <td>googlenet</td>
      <td><a href="./models/googlenet/pt_googlenet_base_img_cls_torchvision.md">pt_googlenet_base_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>96</th>
      <td>gpt2</td>
      <td><a href="./models/gpt2/pt_gpt2_gpt2_text_gen_hf.md">pt_gpt2_gpt2_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>97</th>
      <td>gpt2</td>
      <td><a href="./models/gpt2/pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf.md">pt_gpt2_mnoukhov_gpt2_imdb_sentiment_classifier_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>94 %</td>
      <td>91 %</td>
      <td>6 %</td>
    </tr>
    <tr>
      <th>98</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_125m_clm_hf.md">pt_gptneo_eleutherai_gpt_neo_125m_clm_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>99</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf.md">pt_gptneo_eleutherai_gpt_neo_125m_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>100</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf.md">pt_gptneo_eleutherai_gpt_neo_1_3b_clm_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>101</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf.md">pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>102</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf.md">pt_gptneo_eleutherai_gpt_neo_2_7b_clm_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>90 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>103</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf.md">pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>91 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>104</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm.md">pt_hrnet_hrnet_w18_ms_aug_in1k_pose_estimation_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>105</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_pose_estimation_timm.md">pt_hrnet_hrnet_w18_pose_estimation_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>106</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_pose_estimation_timm.md">pt_hrnet_hrnet_w18_small_pose_estimation_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>107</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr.md">pt_hrnet_hrnet_w18_small_v1_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>108</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr.md">pt_hrnet_hrnet_w18_small_v2_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>109</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm.md">pt_hrnet_hrnet_w18_small_v2_pose_estimation_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>110</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w30_pose_estimation_timm.md">pt_hrnet_hrnet_w30_pose_estimation_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>111</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w18_pose_estimation_osmr.md">pt_hrnet_hrnetv2_w18_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>112</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w30_pose_estimation_osmr.md">pt_hrnet_hrnetv2_w30_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>113</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w32_pose_estimation_osmr.md">pt_hrnet_hrnetv2_w32_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>114</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w40_pose_estimation_osmr.md">pt_hrnet_hrnetv2_w40_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>115</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w44_pose_estimation_osmr.md">pt_hrnet_hrnetv2_w44_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>116</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w48_pose_estimation_osmr.md">pt_hrnet_hrnetv2_w48_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>117</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w64_pose_estimation_osmr.md">pt_hrnet_hrnetv2_w64_pose_estimation_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>118</th>
      <td>inception_v4</td>
      <td><a href="./models/inception_v4/pt_inception_inception_v4_img_cls_timm.md">pt_inception_inception_v4_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>119</th>
      <td>inception_v4</td>
      <td><a href="./models/inception_v4/pt_inception_inception_v4_tf_in1k_img_cls_timm.md">pt_inception_inception_v4_tf_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>120</th>
      <td>inception_v4</td>
      <td><a href="./models/inception_v4/pt_inception_v4_img_cls_osmr.md">pt_inception_v4_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>121</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_huggyllama_llama_7b_clm_hf.md">pt_llama3_huggyllama_llama_7b_clm_hf</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>94 %</td>
      <td>89 %</td>
      <td>6 %</td>
    </tr>
    <tr>
      <th>122</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_huggyllama_llama_7b_seq_cls_hf.md">pt_llama3_huggyllama_llama_7b_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>91 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>123</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_1_8b_clm_hf.md">pt_llama3_meta_llama_llama_3_1_8b_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>91 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>124</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf.md">pt_llama3_meta_llama_llama_3_1_8b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>91 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>125</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf.md">pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>126</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf.md">pt_llama3_meta_llama_llama_3_1_8b_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>127</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_1b_clm_hf.md">pt_llama3_meta_llama_llama_3_2_1b_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>92 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>128</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf.md">pt_llama3_meta_llama_llama_3_2_1b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>92 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>129</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf.md">pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>130</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf.md">pt_llama3_meta_llama_llama_3_2_1b_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>131</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_3b_clm_hf.md">pt_llama3_meta_llama_llama_3_2_3b_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>92 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>132</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf.md">pt_llama3_meta_llama_llama_3_2_3b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>92 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>133</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf.md">pt_llama3_meta_llama_llama_3_2_3b_instruct_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>134</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf.md">pt_llama3_meta_llama_llama_3_2_3b_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>135</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_meta_llama_3_8b_clm_hf.md">pt_llama3_meta_llama_meta_llama_3_8b_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>91 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>136</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf.md">pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>91 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>137</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf.md">pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>138</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf.md">pt_llama3_meta_llama_meta_llama_3_8b_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>139</th>
      <td>llava</td>
      <td><a href="./models/llava/pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf.md">pt_llava_llava_hf_llava_1_5_7b_hf_cond_gen_hf</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>88 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>140</th>
      <td>mistral</td>
      <td><a href="./models/mistral/pt_mistral_mistralai_mistral_7b_v0_1_clm_hf.md">pt_mistral_mistralai_mistral_7b_v0_1_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>141</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_base_img_cls_github.md">pt_mlp_mixer_base_img_cls_github</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>142</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm.md">pt_mlp_mixer_mixer_b16_224_goog_in21k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>143</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_b16_224_img_cls_timm.md">pt_mlp_mixer_mixer_b16_224_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>144</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm.md">pt_mlp_mixer_mixer_b16_224_in21k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>145</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm.md">pt_mlp_mixer_mixer_b16_224_miil_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>146</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm.md">pt_mlp_mixer_mixer_b16_224_miil_in21k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>147</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_l16_224_img_cls_timm.md">pt_mlp_mixer_mixer_l16_224_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>148</th>
      <td>mlp_mixer</td>
      <td><a href="./models/mlp_mixer/pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm.md">pt_mlp_mixer_mixer_l16_224_in21k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>149</th>
      <td>mnist</td>
      <td><a href="./models/mnist/pt_mnist_base_img_cls_github.md">pt_mnist_base_img_cls_github</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>150</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilenet_v1_basic_img_cls_torchvision.md">pt_mobilenet_v1_basic_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>151</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm.md">pt_mobilenet_v1_mobilenetv1_100_ra4_e3600_r224_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>152</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf.md">pt_mobilnet_v1_google_mobilenet_v1_0_75_192_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>153</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf.md">pt_mobilnet_v1_google_mobilenet_v1_1_0_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>154</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_basic_img_cls_torchhub.md">pt_mobilenetv2_basic_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>155</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf.md">pt_mobilenetv2_google_mobilenet_v2_0_35_96_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>92 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>156</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf.md">pt_mobilenetv2_google_mobilenet_v2_0_75_160_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>157</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf.md">pt_mobilenetv2_google_mobilenet_v2_1_0_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>158</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_mobilenet_v2_img_cls_torchvision.md">pt_mobilenetv2_mobilenet_v2_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>159</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_mobilenetv2_100_img_cls_timm.md">pt_mobilenetv2_mobilenetv2_100_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>160</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf.md">pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>94 %</td>
      <td>94 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>161</th>
      <td>mobilenet_v3</td>
      <td><a href="./models/mobilenet_v3/pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub.md">pt_mobilenetv3_mobilenet_v3_large_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>94 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>162</th>
      <td>mobilenet_v3</td>
      <td><a href="./models/mobilenet_v3/pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub.md">pt_mobilenetv3_mobilenet_v3_small_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>94 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>163</th>
      <td>mobilenet_v3</td>
      <td><a href="./models/mobilenet_v3/pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm.md">pt_mobilnetv3_mobilenetv3_large_100_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>164</th>
      <td>mobilenet_v3</td>
      <td><a href="./models/mobilenet_v3/pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm.md">pt_mobilnetv3_mobilenetv3_small_100_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>165</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet101_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet101_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>166</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet152_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet152_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>167</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet18_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet18_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>168</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet34_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet34_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>169</th>
      <td>mobilenet_v3_ssd</td>
      <td><a href="./models/mobilenet_v3_ssd/pt_mobilenetv3_ssd_resnet50_img_cls_torchvision.md">pt_mobilenetv3_ssd_resnet50_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>170</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_1024x320_depth_prediction_torchvision.md">pt_monodepth2_mono_1024x320_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>88 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>171</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_640x192_depth_prediction_torchvision.md">pt_monodepth2_mono_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>90 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>172</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision.md">pt_monodepth2_mono_no_pt_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>90 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>173</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision.md">pt_monodepth2_mono_stereo_1024x320_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>88 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>174</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision.md">pt_monodepth2_mono_stereo_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>90 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>175</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision.md">pt_monodepth2_mono_stereo_no_pt_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>90 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>176</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_stereo_1024x320_depth_prediction_torchvision.md">pt_monodepth2_stereo_1024x320_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>88 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>177</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_stereo_640x192_depth_prediction_torchvision.md">pt_monodepth2_stereo_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>90 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>178</th>
      <td>monodepth2</td>
      <td><a href="./models/monodepth2/pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision.md">pt_monodepth2_stereo_no_pt_640x192_depth_prediction_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>90 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>179</th>
      <td>monodle</td>
      <td><a href="./models/monodle/pt_monodle_base_obj_det_torchvision.md">pt_monodle_base_obj_det_torchvision</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>180</th>
      <td>nanogpt</td>
      <td><a href="./models/nanogpt/pt_nanogpt_financialsupport_nanogpt_text_gen_hf.md">pt_nanogpt_financialsupport_nanogpt_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>181</th>
      <td>nbeats</td>
      <td><a href="./models/nbeats/pt_nbeats_generic_basis_clm_hf.md">pt_nbeats_generic_basis_clm_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>182</th>
      <td>nbeats</td>
      <td><a href="./models/nbeats/pt_nbeats_seasionality_basis_clm_hf.md">pt_nbeats_seasionality_basis_clm_hf</a></td>
      <td>pytorch</td>
      <td>86 %</td>
      <td>86 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>183</th>
      <td>nbeats</td>
      <td><a href="./models/nbeats/pt_nbeats_trend_basis_clm_hf.md">pt_nbeats_trend_basis_clm_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>184</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_clm_hf.md">pt_opt_facebook_opt_125m_clm_hf</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>88 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>185</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_qa_hf.md">pt_opt_facebook_opt_125m_qa_hf</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>88 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>186</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_seq_cls_hf.md">pt_opt_facebook_opt_125m_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>85 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>187</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_1_3b_clm_hf.md">pt_opt_facebook_opt_1_3b_clm_hf</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>88 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>188</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_1_3b_qa_hf.md">pt_opt_facebook_opt_1_3b_qa_hf</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>88 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>189</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_1_3b_seq_cls_hf.md">pt_opt_facebook_opt_1_3b_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>85 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>190</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_350m_clm_hf.md">pt_opt_facebook_opt_350m_clm_hf</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>89 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>191</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_350m_qa_hf.md">pt_opt_facebook_opt_350m_qa_hf</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>89 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>192</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_350m_seq_cls_hf.md">pt_opt_facebook_opt_350m_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>86 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>193</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_language_perceiver_mlm_hf.md">pt_perceiverio_deepmind_language_perceiver_mlm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>194</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf.md">pt_perceiverio_deepmind_vision_perceiver_conv_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>195</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf.md">pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>196</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf.md">pt_perceiverio_deepmind_vision_perceiver_learned_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>197</th>
      <td>phi1_5_pt</td>
      <td><a href="./models/phi1_5_pt/pt_phi1_5_microsoft_phi_1_5_clm_hf.md">pt_phi1_5_microsoft_phi_1_5_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>198</th>
      <td>phi1_5_pt</td>
      <td><a href="./models/phi1_5_pt/pt_phi1_5_microsoft_phi_1_5_seq_cls_hf.md">pt_phi1_5_microsoft_phi_1_5_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>199</th>
      <td>phi1_5_pt</td>
      <td><a href="./models/phi1_5_pt/pt_phi1_5_microsoft_phi_1_5_token_cls_hf.md">pt_phi1_5_microsoft_phi_1_5_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>200</th>
      <td>phi1_pt</td>
      <td><a href="./models/phi1_pt/pt_phi1_microsoft_phi_1_clm_hf.md">pt_phi1_microsoft_phi_1_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>201</th>
      <td>phi1_pt</td>
      <td><a href="./models/phi1_pt/pt_phi1_microsoft_phi_1_seq_cls_hf.md">pt_phi1_microsoft_phi_1_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>202</th>
      <td>phi1_pt</td>
      <td><a href="./models/phi1_pt/pt_phi1_microsoft_phi_1_token_cls_hf.md">pt_phi1_microsoft_phi_1_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>203</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_clm_hf.md">pt_phi2_microsoft_phi_2_clm_hf</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>94 %</td>
      <td>89 %</td>
      <td>6 %</td>
    </tr>
    <tr>
      <th>204</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_pytdml_clm_hf.md">pt_phi2_microsoft_phi_2_pytdml_clm_hf</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>94 %</td>
      <td>89 %</td>
      <td>6 %</td>
    </tr>
    <tr>
      <th>205</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf.md">pt_phi2_microsoft_phi_2_pytdml_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>206</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_pytdml_token_cls_hf.md">pt_phi2_microsoft_phi_2_pytdml_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>207</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_seq_cls_hf.md">pt_phi2_microsoft_phi_2_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>208</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_token_cls_hf.md">pt_phi2_microsoft_phi_2_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>209</th>
      <td>phi3</td>
      <td><a href="./models/phi3/pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf.md">pt_phi3_microsoft_phi_3_mini_4k_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>91 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>210</th>
      <td>phi3</td>
      <td><a href="./models/phi3/pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf.md">pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>211</th>
      <td>phi3</td>
      <td><a href="./models/phi3/pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf.md">pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>212</th>
      <td>phi3_5</td>
      <td><a href="./models/phi3_5/pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf.md">pt_phi3_5_microsoft_phi_3_5_mini_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>94 %</td>
      <td>3 %</td>
    </tr>
    <tr>
      <th>213</th>
      <td>phi4</td>
      <td><a href="./models/phi4/pt_phi4_microsoft_phi_4_clm_hf.md">pt_phi4_microsoft_phi_4_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>214</th>
      <td>phi4</td>
      <td><a href="./models/phi4/pt_phi4_microsoft_phi_4_seq_cls_hf.md">pt_phi4_microsoft_phi_4_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>215</th>
      <td>phi4</td>
      <td><a href="./models/phi4/pt_phi4_microsoft_phi_4_token_cls_hf.md">pt_phi4_microsoft_phi_4_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>216</th>
      <td>qwen</td>
      <td><a href="./models/qwen/pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf.md">pt_qwen1_5_qwen_qwen1_5_0_5b_chat_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>217</th>
      <td>qwen</td>
      <td><a href="./models/qwen/pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf.md">pt_qwen1_5_qwen_qwen1_5_0_5b_clm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>218</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf.md">pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>219</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf.md">pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>220</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf.md">pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>221</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf.md">pt_qwen_coder_qwen_qwen2_5_coder_3b_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>222</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf.md">pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>223</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf.md">pt_qwen_coder_qwen_qwen2_5_coder_7b_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>224</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf.md">pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>225</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_0_5b_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>226</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>227</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_1_5b_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>228</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>229</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_3b_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_3b_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>230</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>231</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_7b_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_7b_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>232</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf.md">pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>233</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_7b_token_cls_hf.md">pt_qwen_v2_qwen_qwen2_7b_token_cls_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>234</th>
      <td>rcnn</td>
      <td><a href="./models/rcnn/pt_rcnn_base_obj_det_torchvision_rect_0.md">pt_rcnn_base_obj_det_torchvision_rect_0</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>235</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_facebook_regnet_y_040_img_cls_hf.md">pt_regnet_facebook_regnet_y_040_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>236</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_x_16gf_img_cls_torchvision.md">pt_regnet_regnet_x_16gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>237</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_x_1_6gf_img_cls_torchvision.md">pt_regnet_regnet_x_1_6gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>238</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_x_32gf_img_cls_torchvision.md">pt_regnet_regnet_x_32gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>239</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_x_3_2gf_img_cls_torchvision.md">pt_regnet_regnet_x_3_2gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>240</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_x_400mf_img_cls_torchvision.md">pt_regnet_regnet_x_400mf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>241</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_x_800mf_img_cls_torchvision.md">pt_regnet_regnet_x_800mf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>242</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_x_8gf_img_cls_torchvision.md">pt_regnet_regnet_x_8gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>243</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_128gf_img_cls_torchvision.md">pt_regnet_regnet_y_128gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>244</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_16gf_img_cls_torchvision.md">pt_regnet_regnet_y_16gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>245</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_1_6gf_img_cls_torchvision.md">pt_regnet_regnet_y_1_6gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>246</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_32gf_img_cls_torchvision.md">pt_regnet_regnet_y_32gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>247</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_3_2gf_img_cls_torchvision.md">pt_regnet_regnet_y_3_2gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>248</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_400mf_img_cls_torchvision.md">pt_regnet_regnet_y_400mf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>249</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_800mf_img_cls_torchvision.md">pt_regnet_regnet_y_800mf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>250</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_regnet_y_8gf_img_cls_torchvision.md">pt_regnet_regnet_y_8gf_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>251</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_50_img_cls_hf.md">pt_resnet_50_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>252</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_50_img_cls_timm.md">pt_resnet_50_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>253</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet101_img_cls_torchvision.md">pt_resnet_resnet101_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>254</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet152_img_cls_torchvision.md">pt_resnet_resnet152_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>255</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet18_img_cls_torchvision.md">pt_resnet_resnet18_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>256</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet34_img_cls_torchvision.md">pt_resnet_resnet34_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>257</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_resnet50_img_cls_torchvision.md">pt_resnet_resnet50_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>258</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext101_32x8d_img_cls_torchhub.md">pt_resnext_resnext101_32x8d_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>259</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub.md">pt_resnext_resnext101_32x8d_wsl_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>260</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext101_64x4d_img_cls_osmr.md">pt_resnext_resnext101_64x4d_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>261</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext14_32x4d_img_cls_osmr.md">pt_resnext_resnext14_32x4d_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>262</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext26_32x4d_img_cls_osmr.md">pt_resnext_resnext26_32x4d_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>263</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext50_32x4d_img_cls_osmr.md">pt_resnext_resnext50_32x4d_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>264</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext50_32x4d_img_cls_torchhub.md">pt_resnext_resnext50_32x4d_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>265</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn101fpn_obj_det_hf.md">pt_retinanet_retinanet_rn101fpn_obj_det_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>266</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn152fpn_obj_det_hf.md">pt_retinanet_retinanet_rn152fpn_obj_det_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>267</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn18fpn_obj_det_hf.md">pt_retinanet_retinanet_rn18fpn_obj_det_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>268</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn34fpn_obj_det_hf.md">pt_retinanet_retinanet_rn34fpn_obj_det_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>269</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn50fpn_obj_det_hf.md">pt_retinanet_retinanet_rn50fpn_obj_det_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>270</th>
      <td>roberta</td>
      <td><a href="./models/roberta/pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf.md">pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>271</th>
      <td>roberta</td>
      <td><a href="./models/roberta/pt_roberta_xlm_roberta_base_mlm_hf.md">pt_roberta_xlm_roberta_base_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>94 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>272</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b0_img_cls_hf.md">pt_segformer_nvidia_mit_b0_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>273</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b1_img_cls_hf.md">pt_segformer_nvidia_mit_b1_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>274</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b2_img_cls_hf.md">pt_segformer_nvidia_mit_b2_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>275</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b3_img_cls_hf.md">pt_segformer_nvidia_mit_b3_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>276</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b4_img_cls_hf.md">pt_segformer_nvidia_mit_b4_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>277</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b5_img_cls_hf.md">pt_segformer_nvidia_mit_b5_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>278</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf.md">pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>279</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf.md">pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>280</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf.md">pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>281</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf.md">pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>282</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf.md">pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>283</th>
      <td>squeezebert</td>
      <td><a href="./models/squeezebert/pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf.md">pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>284</th>
      <td>ssd300_resnet50</td>
      <td><a href="./models/ssd300_resnet50/pt_ssd300_resnet50_base_img_cls_torchhub.md">pt_ssd300_resnet50_base_img_cls_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>285</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_large_music_generation_hf.md">pt_stereo_facebook_musicgen_large_music_generation_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>92 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>286</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_medium_music_generation_hf.md">pt_stereo_facebook_musicgen_medium_music_generation_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>93 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>287</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_small_music_generation_hf.md">pt_stereo_facebook_musicgen_small_music_generation_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>93 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>288</th>
      <td>swin</td>
      <td><a href="./models/swin/pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf.md">pt_swin_microsoft_swin_tiny_patch4_window7_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>289</th>
      <td>swin</td>
      <td><a href="./models/swin/pt_swin_swin_b_img_cls_torchvision.md">pt_swin_swin_b_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>290</th>
      <td>swin</td>
      <td><a href="./models/swin/pt_swin_swin_s_img_cls_torchvision.md">pt_swin_swin_s_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>291</th>
      <td>swin</td>
      <td><a href="./models/swin/pt_swin_swin_t_img_cls_torchvision.md">pt_swin_swin_t_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>292</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_google_flan_t5_base_text_gen_hf.md">pt_t5_google_flan_t5_base_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>293</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_google_flan_t5_large_text_gen_hf.md">pt_t5_google_flan_t5_large_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>294</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_google_flan_t5_small_text_gen_hf.md">pt_t5_google_flan_t5_small_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>295</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_base_text_gen_hf.md">pt_t5_t5_base_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>296</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_large_text_gen_hf.md">pt_t5_t5_large_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>297</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_small_text_gen_hf.md">pt_t5_t5_small_text_gen_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>298</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_base_img_seg_torchhub.md">pt_unet_base_img_seg_torchhub</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>299</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_carvana_base_img_seg_github.md">pt_unet_carvana_base_img_seg_github</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>91 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <th>300</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_cityscape_img_seg_osmr.md">pt_unet_cityscape_img_seg_osmr</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>92 %</td>
      <td>92 %</td>
      <td>8 %</td>
    </tr>
    <tr>
      <th>301</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_qubvel_img_seg_torchhub.md">pt_unet_qubvel_img_seg_torchhub</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>302</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_19_obj_det_hf.md">pt_vgg_19_obj_det_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>303</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_bn_vgg19_obj_det_osmr.md">pt_vgg_bn_vgg19_obj_det_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>304</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_bn_vgg19b_obj_det_osmr.md">pt_vgg_bn_vgg19b_obj_det_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>305</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg11_bn_img_cls_torchvision.md">pt_vgg_vgg11_bn_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>306</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg11_img_cls_torchvision.md">pt_vgg_vgg11_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>307</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg11_obj_det_osmr.md">pt_vgg_vgg11_obj_det_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>308</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg13_bn_img_cls_torchvision.md">pt_vgg_vgg13_bn_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>309</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg13_img_cls_torchvision.md">pt_vgg_vgg13_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>310</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg13_obj_det_osmr.md">pt_vgg_vgg13_obj_det_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>311</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg16_bn_img_cls_torchvision.md">pt_vgg_vgg16_bn_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>312</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg16_img_cls_torchvision.md">pt_vgg_vgg16_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>313</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg16_obj_det_osmr.md">pt_vgg_vgg16_obj_det_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>314</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg19_bn_obj_det_timm.md">pt_vgg_vgg19_bn_obj_det_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>315</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg19_bn_obj_det_torchhub.md">pt_vgg_vgg19_bn_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>316</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg19_img_cls_torchvision.md">pt_vgg_vgg19_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>317</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg19_obj_det_osmr.md">pt_vgg_vgg19_obj_det_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>318</th>
      <td>vilt</td>
      <td><a href="./models/vilt/pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf.md">pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>319</th>
      <td>vilt</td>
      <td><a href="./models/vilt/pt_vilt_dandelin_vilt_b32_mlm_mlm_hf.md">pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>320</th>
      <td>vit</td>
      <td><a href="./models/vit/pt_vit_google_vit_base_patch16_224_img_cls_hf.md">pt_vit_google_vit_base_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>321</th>
      <td>vit</td>
      <td><a href="./models/vit/pt_vit_google_vit_large_patch16_224_img_cls_hf.md">pt_vit_google_vit_large_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>322</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub.md">pt_vovnet_ese_vovnet19b_dw_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>323</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub.md">pt_vovnet_ese_vovnet19b_dw_ra_in1k_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>324</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_ese_vovnet39b_obj_det_torchhub.md">pt_vovnet_ese_vovnet39b_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>325</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_v1_vovnet39_obj_det_torchhub.md">pt_vovnet_v1_vovnet39_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>326</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_vovnet27s_img_cls_osmr.md">pt_vovnet_vovnet27s_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>327</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_vovnet39_img_cls_osmr.md">pt_vovnet_vovnet39_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>328</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_vovnet57_img_cls_osmr.md">pt_vovnet_vovnet57_img_cls_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>329</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_vovnet_v1_57_obj_det_torchhub.md">pt_vovnet_vovnet_v1_57_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>330</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_base_speech_recognition_hf.md">pt_whisper_openai_whisper_base_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>331</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_large_speech_recognition_hf.md">pt_whisper_openai_whisper_large_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>332</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_medium_speech_recognition_hf.md">pt_whisper_openai_whisper_medium_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <th>333</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_small_speech_recognition_hf.md">pt_whisper_openai_whisper_small_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>334</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_tiny_speech_recognition_hf.md">pt_whisper_openai_whisper_tiny_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>335</th>
      <td>whisper_large_v3</td>
      <td><a href="./models/whisper_large_v3/pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf.md">pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>336</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet101_2_img_cls_timm.md">pt_wideresnet_wide_resnet101_2_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>337</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet101_2_img_cls_torchvision.md">pt_wideresnet_wide_resnet101_2_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>338</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet50_2_img_cls_timm.md">pt_wideresnet_wide_resnet50_2_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>339</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet50_2_img_cls_torchvision.md">pt_wideresnet_wide_resnet50_2_img_cls_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>340</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception41_img_cls_timm.md">pt_xception_xception41_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>12 %</td>
    </tr>
    <tr>
      <th>341</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception65_img_cls_timm.md">pt_xception_xception65_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>85 %</td>
      <td>85 %</td>
      <td>85 %</td>
      <td>15 %</td>
    </tr>
    <tr>
      <th>342</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception71_img_cls_timm.md">pt_xception_xception71_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>85 %</td>
      <td>85 %</td>
      <td>85 %</td>
      <td>15 %</td>
    </tr>
    <tr>
      <th>343</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception71_tf_in1k_img_cls_timm.md">pt_xception_xception71_tf_in1k_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>85 %</td>
      <td>85 %</td>
      <td>85 %</td>
      <td>15 %</td>
    </tr>
    <tr>
      <th>344</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception_img_cls_timm.md">pt_xception_xception_img_cls_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>345</th>
      <td>xglm</td>
      <td><a href="./models/xglm/pt_xglm_facebook_xglm_1_7b_clm_hf.md">pt_xglm_facebook_xglm_1_7b_clm_hf</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>91 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>346</th>
      <td>xglm</td>
      <td><a href="./models/xglm/pt_xglm_facebook_xglm_564m_clm_hf.md">pt_xglm_facebook_xglm_564m_clm_hf</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>91 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>347</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5l_img_cls_torchhub_320x320.md">pt_yolo_v5_yolov5l_img_cls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>348</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5l_img_cls_torchhub_480x480.md">pt_yolo_v5_yolov5l_img_cls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>349</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5l_img_cls_torchhub_640x640.md">pt_yolo_v5_yolov5l_img_cls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>350</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5m_img_cls_torchhub_320x320.md">pt_yolo_v5_yolov5m_img_cls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>351</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5m_img_cls_torchhub_480x480.md">pt_yolo_v5_yolov5m_img_cls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>352</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5m_img_cls_torchhub_640x640.md">pt_yolo_v5_yolov5m_img_cls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>353</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5n_img_cls_torchhub_320x320.md">pt_yolo_v5_yolov5n_img_cls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>354</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5n_img_cls_torchhub_480x480.md">pt_yolo_v5_yolov5n_img_cls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>355</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5n_img_cls_torchhub_640x640.md">pt_yolo_v5_yolov5n_img_cls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>356</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280.md">pt_yolo_v5_yolov5s_img_cls_torchhub_1280x1280</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>357</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_img_cls_torchhub_320x320.md">pt_yolo_v5_yolov5s_img_cls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>358</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_img_cls_torchhub_480x480.md">pt_yolo_v5_yolov5s_img_cls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>359</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_img_cls_torchhub_640x640.md">pt_yolo_v5_yolov5s_img_cls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>360</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5x_img_cls_torchhub_320x320.md">pt_yolo_v5_yolov5x_img_cls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>361</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5x_img_cls_torchhub_480x480.md">pt_yolo_v5_yolov5x_img_cls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>362</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5x_img_cls_torchhub_640x640.md">pt_yolo_v5_yolov5x_img_cls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>363</th>
      <td>yolo_v6</td>
      <td><a href="./models/yolo_v6/pt_yolo_v6_yolov6l_obj_det_torchhub.md">pt_yolo_v6_yolov6l_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>364</th>
      <td>yolo_v6</td>
      <td><a href="./models/yolo_v6/pt_yolo_v6_yolov6m_obj_det_torchhub.md">pt_yolo_v6_yolov6m_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>365</th>
      <td>yolo_v6</td>
      <td><a href="./models/yolo_v6/pt_yolo_v6_yolov6n_obj_det_torchhub.md">pt_yolo_v6_yolov6n_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>366</th>
      <td>yolo_v6</td>
      <td><a href="./models/yolo_v6/pt_yolo_v6_yolov6s_obj_det_torchhub.md">pt_yolo_v6_yolov6s_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>367</th>
      <td>yolo_v8</td>
      <td><a href="./models/yolo_v8/pt_yolov8_default_obj_det_github.md">pt_yolov8_default_obj_det_github</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>368</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_darknet_obj_det_torchhub.md">pt_yolox_yolox_darknet_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>369</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_l_obj_det_torchhub.md">pt_yolox_yolox_l_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>370</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_m_obj_det_torchhub.md">pt_yolox_yolox_m_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>371</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_nano_obj_det_torchhub.md">pt_yolox_yolox_nano_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>372</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_s_obj_det_torchhub.md">pt_yolox_yolox_s_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>373</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_tiny_obj_det_torchhub.md">pt_yolox_yolox_tiny_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>374</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_x_obj_det_torchhub.md">pt_yolox_yolox_x_obj_det_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
  </tbody>
</table>
