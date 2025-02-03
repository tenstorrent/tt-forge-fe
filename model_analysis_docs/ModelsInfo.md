<h1>List of models and current compiler support rates</h1>
<p><b>Last updated date and time(in GMT) :</b> Monday, 27 Jan 2025 08:49:10 PM</p><p><b>Commit Id :</b> <a href="https://github.com/tenstorrent/tt-forge-fe/commit/03ef10f4072e197eb0485a819a69ef5ae3149eea">03ef10f4072e197eb0485a819a69ef5ae3149eea</a></p><p><b>Note:</b> For detailed insights into compiler failures and their effects on models, please refer to the <a href="./stats/compiler_analysis_report.md">compiler_analysis_report.md</a>.</p><table border="1" class="dataframe">
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
      <td><a href="./models/albert/pt_albert_base_v1_mlm.md">pt_albert_base_v1_mlm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>2</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v1_token_cls.md">pt_albert_base_v1_token_cls</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v2_mlm.md">pt_albert_base_v2_mlm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_base_v2_token_cls.md">pt_albert_base_v2_token_cls</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>5</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_large_v1_mlm.md">pt_albert_large_v1_mlm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>6</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_large_v1_token_cls.md">pt_albert_large_v1_token_cls</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>7</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_large_v2_mlm.md">pt_albert_large_v2_mlm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>8</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_large_v2_token_cls.md">pt_albert_large_v2_token_cls</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>9</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xlarge_v1_mlm.md">pt_albert_xlarge_v1_mlm</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>10</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xlarge_v1_token_cls.md">pt_albert_xlarge_v1_token_cls</a></td>
      <td>pytorch</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>11</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xlarge_v2_mlm.md">pt_albert_xlarge_v2_mlm</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>12</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xlarge_v2_token_cls.md">pt_albert_xlarge_v2_token_cls</a></td>
      <td>pytorch</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>13</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xxlarge_v1_mlm.md">pt_albert_xxlarge_v1_mlm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>14</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xxlarge_v1_token_cls.md">pt_albert_xxlarge_v1_token_cls</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>15</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xxlarge_v2_mlm.md">pt_albert_xxlarge_v2_mlm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>16</th>
      <td>albert</td>
      <td><a href="./models/albert/pt_albert_xxlarge_v2_token_cls.md">pt_albert_xxlarge_v2_token_cls</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>17</th>
      <td>alexnet</td>
      <td><a href="./models/alexnet/pt_alexnet_alexnet_torchhub.md">pt_alexnet_alexnet_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>18</th>
      <td>alexnet</td>
      <td><a href="./models/alexnet/pt_alexnet_base_osmr.md">pt_alexnet_base_osmr</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>92 %</td>
      <td>83 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <th>19</th>
      <td>autoencoder</td>
      <td><a href="./models/autoencoder/pt_autoencoder_conv.md">pt_autoencoder_conv</a></td>
      <td>pytorch</td>
      <td>84 %</td>
      <td>84 %</td>
      <td>84 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <th>20</th>
      <td>autoencoder</td>
      <td><a href="./models/autoencoder/pt_autoencoder_linear.md">pt_autoencoder_linear</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>21</th>
      <td>bart</td>
      <td><a href="./models/bart/pt_bart_facebook_bart_large_mnli_seq_cls.md">pt_bart_facebook_bart_large_mnli_seq_cls</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>22</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_bert_base_uncased_mlm.md">pt_bert_bert_base_uncased_mlm</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>23</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa.md">pt_bert_bert_large_cased_whole_word_masking_finetuned_squad_qa</a></td>
      <td>pytorch</td>
      <td>84 %</td>
      <td>84 %</td>
      <td>84 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>24</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls.md">pt_bert_dbmdz_bert_large_cased_finetuned_conll03_english_token_cls</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>25</th>
      <td>bert</td>
      <td><a href="./models/bert/pt_bert_textattack_bert_base_uncased_sst_2_seq_cls.md">pt_bert_textattack_bert_base_uncased_sst_2_seq_cls</a></td>
      <td>pytorch</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>26</th>
      <td>clip</td>
      <td><a href="./models/clip/pt_clip_openai_clip_vit_base_patch32_text.md">pt_clip_openai_clip_vit_base_patch32_text</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>27</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_mono_clm.md">pt_codegen_salesforce_codegen_350m_mono_clm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>28</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_multi_clm.md">pt_codegen_salesforce_codegen_350m_multi_clm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>29</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_nl_clm.md">pt_codegen_salesforce_codegen_350m_nl_clm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>30</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_base_distilled_patch16_224_img_cls.md">pt_deit_facebook_deit_base_distilled_patch16_224_img_cls</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>31</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_base_patch16_224_img_cls.md">pt_deit_facebook_deit_base_patch16_224_img_cls</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>32</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_small_patch16_224_img_cls.md">pt_deit_facebook_deit_small_patch16_224_img_cls</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>33</th>
      <td>deit</td>
      <td><a href="./models/deit/pt_deit_facebook_deit_tiny_patch16_224_img_cls.md">pt_deit_facebook_deit_tiny_patch16_224_img_cls</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>34</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet121.md">pt_densenet_densenet121</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>35</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet161.md">pt_densenet_densenet161</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>36</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet169.md">pt_densenet_densenet169</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>37</th>
      <td>densenet</td>
      <td><a href="./models/densenet/pt_densenet_densenet201.md">pt_densenet_densenet201</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>38</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls.md">pt_distilbert_davlan_distilbert_base_multilingual_cased_ner_hrl_token_cls</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>88 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>39</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_cased_distilled_squad_qa.md">pt_distilbert_distilbert_base_cased_distilled_squad_qa</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>85 %</td>
      <td>84 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>40</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_cased_mlm.md">pt_distilbert_distilbert_base_cased_mlm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>88 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>41</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_multilingual_cased_mlm.md">pt_distilbert_distilbert_base_multilingual_cased_mlm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>88 %</td>
      <td>84 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>42</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls.md">pt_distilbert_distilbert_base_uncased_finetuned_sst_2_english_seq_cls</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>89 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>43</th>
      <td>distilbert</td>
      <td><a href="./models/distilbert/pt_distilbert_distilbert_base_uncased_mlm.md">pt_distilbert_distilbert_base_uncased_mlm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>88 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>44</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla102.md">pt_dla_dla102</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>45</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla102x.md">pt_dla_dla102x</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>46</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla102x2.md">pt_dla_dla102x2</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>47</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla169.md">pt_dla_dla169</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>48</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla34.md">pt_dla_dla34</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>49</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla46_c.md">pt_dla_dla46_c</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>50</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla46x_c.md">pt_dla_dla46x_c</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>51</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla60.md">pt_dla_dla60</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>52</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla60x.md">pt_dla_dla60x</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>53</th>
      <td>dla</td>
      <td><a href="./models/dla/pt_dla_dla60x_c.md">pt_dla_dla60x_c</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>54</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder.md">pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>55</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder.md">pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>56</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_question_encoder_multiset_base_question_encoder.md">pt_dpr_facebook_dpr_question_encoder_multiset_base_question_encoder</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>57</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_question_encoder_single_nq_base_question_encoder.md">pt_dpr_facebook_dpr_question_encoder_single_nq_base_question_encoder</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>58</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_reader_multiset_base_reader.md">pt_dpr_facebook_dpr_reader_multiset_base_reader</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>59</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_reader_single_nq_base_reader.md">pt_dpr_facebook_dpr_reader_single_nq_base_reader</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>60</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b0_timm.md">pt_efficientnet_efficientnet_b0_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>61</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b0_torchvision.md">pt_efficientnet_efficientnet_b0_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>62</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b4_timm.md">pt_efficientnet_efficientnet_b4_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>63</th>
      <td>efficientnet</td>
      <td><a href="./models/efficientnet/pt_efficientnet_efficientnet_b4_torchvision.md">pt_efficientnet_efficientnet_b4_torchvision</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>64</th>
      <td>falcon</td>
      <td><a href="./models/falcon/pt_falcon_tiiuae_falcon_7b_instruct.md">pt_falcon_tiiuae_falcon_7b_instruct</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>94 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>65</th>
      <td>fpn</td>
      <td><a href="./models/fpn/pt_fpn_base_torchvision.md">pt_fpn_base_torchvision</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>66</th>
      <td>fuyu_8b</td>
      <td><a href="./models/fuyu_8b/pt_fuyu_adept_fuyu_8b.md">pt_fuyu_adept_fuyu_8b</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>67</th>
      <td>gemma_2b</td>
      <td><a href="./models/gemma_2b/pt_gemma_google_gemma_2b.md">pt_gemma_google_gemma_2b</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>92 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>68</th>
      <td>ghostnet</td>
      <td><a href="./models/ghostnet/pt_ghostnet_ghostnet_100_timm.md">pt_ghostnet_ghostnet_100_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>69</th>
      <td>googlenet</td>
      <td><a href="./models/googlenet/pt_googlenet_base.md">pt_googlenet_base</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>70</th>
      <td>gpt2</td>
      <td><a href="./models/gpt2/pt_gpt2_gpt2_text_gen.md">pt_gpt2_gpt2_text_gen</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>88 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>71</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_125m_clm.md">pt_gptneo_eleutherai_gpt_neo_125m_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>72</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_125m_seq_cls.md">pt_gptneo_eleutherai_gpt_neo_125m_seq_cls</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>90 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>73</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_1_3b_clm.md">pt_gptneo_eleutherai_gpt_neo_1_3b_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>74</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls.md">pt_gptneo_eleutherai_gpt_neo_1_3b_seq_cls</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>90 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>75</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_2_7b_clm.md">pt_gptneo_eleutherai_gpt_neo_2_7b_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>84 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>76</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls.md">pt_gptneo_eleutherai_gpt_neo_2_7b_seq_cls</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>90 %</td>
      <td>85 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>77</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_timm.md">pt_hrnet_hrnet_w18_small_timm</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>78</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_v1_osmr.md">pt_hrnet_hrnet_w18_small_v1_osmr</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>79</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_v2_osmr.md">pt_hrnet_hrnet_w18_small_v2_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>80</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_small_v2_timm.md">pt_hrnet_hrnet_w18_small_v2_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>81</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w18_timm.md">pt_hrnet_hrnet_w18_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>82</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w30_timm.md">pt_hrnet_hrnet_w30_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>83</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w32_timm.md">pt_hrnet_hrnet_w32_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>84</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w40_timm.md">pt_hrnet_hrnet_w40_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>85</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w44_timm.md">pt_hrnet_hrnet_w44_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>86</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w48_timm.md">pt_hrnet_hrnet_w48_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>87</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnet_w64_timm.md">pt_hrnet_hrnet_w64_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>88</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w18_osmr.md">pt_hrnet_hrnetv2_w18_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>89</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w30_osmr.md">pt_hrnet_hrnetv2_w30_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>90</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w32_osmr.md">pt_hrnet_hrnetv2_w32_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>91</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w40_osmr.md">pt_hrnet_hrnetv2_w40_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>92</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w44_osmr.md">pt_hrnet_hrnetv2_w44_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>93</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w48_osmr.md">pt_hrnet_hrnetv2_w48_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>94</th>
      <td>hrnet</td>
      <td><a href="./models/hrnet/pt_hrnet_hrnetv2_w64_osmr.md">pt_hrnet_hrnetv2_w64_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>95</th>
      <td>inception_v4</td>
      <td><a href="./models/inception_v4/pt_inception_v4_osmr.md">pt_inception_v4_osmr</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>96</th>
      <td>inception_v4</td>
      <td><a href="./models/inception_v4/pt_inception_v4_timm.md">pt_inception_v4_timm</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>97</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_1_8b_clm.md">pt_llama3_meta_llama_llama_3_1_8b_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>98</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_1_8b_instruct_clm.md">pt_llama3_meta_llama_llama_3_1_8b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>99</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls.md">pt_llama3_meta_llama_llama_3_1_8b_instruct_seq_cls</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>100</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_1_8b_seq_cls.md">pt_llama3_meta_llama_llama_3_1_8b_seq_cls</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>101</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_1b_clm.md">pt_llama3_meta_llama_llama_3_2_1b_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>102</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_1b_instruct_clm.md">pt_llama3_meta_llama_llama_3_2_1b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>103</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls.md">pt_llama3_meta_llama_llama_3_2_1b_instruct_seq_cls</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>104</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_llama_3_2_1b_seq_cls.md">pt_llama3_meta_llama_llama_3_2_1b_seq_cls</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>105</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_meta_llama_3_8b_clm.md">pt_llama3_meta_llama_meta_llama_3_8b_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>106</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm.md">pt_llama3_meta_llama_meta_llama_3_8b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>107</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls.md">pt_llama3_meta_llama_meta_llama_3_8b_instruct_seq_cls</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>108</th>
      <td>llama3</td>
      <td><a href="./models/llama3/pt_llama3_meta_llama_meta_llama_3_8b_seq_cls.md">pt_llama3_meta_llama_meta_llama_3_8b_seq_cls</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>109</th>
      <td>mistral</td>
      <td><a href="./models/mistral/pt_mistral_mistralai_mistral_7b_v0_1.md">pt_mistral_mistralai_mistral_7b_v0_1</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>110</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilenet_v1_basic.md">pt_mobilenet_v1_basic</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>111</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf.md">pt_mobilnet_v1_google_mobilenet_v1_0_75_192_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>90 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>112</th>
      <td>mobilenet_v1</td>
      <td><a href="./models/mobilenet_v1/pt_mobilnet_v1_google_mobilenet_v1_1_0_224_hf.md">pt_mobilnet_v1_google_mobilenet_v1_1_0_224_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>113</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_basic_torchhub.md">pt_mobilenetv2_basic_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>94 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>114</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf.md">pt_mobilenetv2_google_mobilenet_v2_0_35_96_hf</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>92 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>115</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf.md">pt_mobilenetv2_google_mobilenet_v2_0_75_160_hf</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>116</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_google_mobilenet_v2_1_0_224_hf.md">pt_mobilenetv2_google_mobilenet_v2_1_0_224_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>117</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilenetv2_mobilenetv2_100_timm.md">pt_mobilenetv2_mobilenetv2_100_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>118</th>
      <td>mobilenet_v2</td>
      <td><a href="./models/mobilenet_v2/pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf.md">pt_mobilnetv2_google_deeplabv3_mobilenet_v2_1_0_513_hf</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>94 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>119</th>
      <td>mobilenet_v3</td>
      <td><a href="./models/mobilenet_v3/pt_mobilnetv3_mobilenetv3_large_100_timm.md">pt_mobilnetv3_mobilenetv3_large_100_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>120</th>
      <td>mobilenet_v3</td>
      <td><a href="./models/mobilenet_v3/pt_mobilnetv3_mobilenetv3_small_100_timm.md">pt_mobilnetv3_mobilenetv3_small_100_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>94 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>121</th>
      <td>monodle</td>
      <td><a href="./models/monodle/pt_monodle_base.md">pt_monodle_base</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>122</th>
      <td>nbeats</td>
      <td><a href="./models/nbeats/pt_nbeats_generic_basis.md">pt_nbeats_generic_basis</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>123</th>
      <td>nbeats</td>
      <td><a href="./models/nbeats/pt_nbeats_seasionality_basis.md">pt_nbeats_seasionality_basis</a></td>
      <td>pytorch</td>
      <td>86 %</td>
      <td>86 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>124</th>
      <td>nbeats</td>
      <td><a href="./models/nbeats/pt_nbeats_trend_basis.md">pt_nbeats_trend_basis</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>125</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_clm.md">pt_opt_facebook_opt_125m_clm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>126</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_qa.md">pt_opt_facebook_opt_125m_qa</a></td>
      <td>pytorch</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>127</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_seq_cls.md">pt_opt_facebook_opt_125m_seq_cls</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>83 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>128</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_1_3b_clm.md">pt_opt_facebook_opt_1_3b_clm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>129</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_1_3b_qa.md">pt_opt_facebook_opt_1_3b_qa</a></td>
      <td>pytorch</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>85 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>130</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_1_3b_seq_cls.md">pt_opt_facebook_opt_1_3b_seq_cls</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>83 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>131</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_350m_clm.md">pt_opt_facebook_opt_350m_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>132</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_350m_qa.md">pt_opt_facebook_opt_350m_qa</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>133</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_350m_seq_cls.md">pt_opt_facebook_opt_350m_seq_cls</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>84 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>134</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_vision_perceiver_conv_img_cls.md">pt_perceiverio_deepmind_vision_perceiver_conv_img_cls</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>135</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls.md">pt_perceiverio_deepmind_vision_perceiver_fourier_img_cls</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>136</th>
      <td>perceiverio</td>
      <td><a href="./models/perceiverio/pt_perceiverio_deepmind_vision_perceiver_learned_img_cls.md">pt_perceiverio_deepmind_vision_perceiver_learned_img_cls</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>137</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_clm.md">pt_phi2_microsoft_phi_2_clm</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>138</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_pytdml_clm.md">pt_phi2_microsoft_phi_2_pytdml_clm</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>139</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_pytdml_seq_cls.md">pt_phi2_microsoft_phi_2_pytdml_seq_cls</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>140</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_pytdml_token_cls.md">pt_phi2_microsoft_phi_2_pytdml_token_cls</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>141</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_seq_cls.md">pt_phi2_microsoft_phi_2_seq_cls</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>142</th>
      <td>phi2</td>
      <td><a href="./models/phi2/pt_phi2_microsoft_phi_2_token_cls.md">pt_phi2_microsoft_phi_2_token_cls</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>143</th>
      <td>phi3</td>
      <td><a href="./models/phi3/pt_phi3_microsoft_phi_3_mini_4k_instruct_clm.md">pt_phi3_microsoft_phi_3_mini_4k_instruct_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>144</th>
      <td>phi3</td>
      <td><a href="./models/phi3/pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls.md">pt_phi3_microsoft_phi_3_mini_4k_instruct_seq_cls</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>94 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>145</th>
      <td>phi3</td>
      <td><a href="./models/phi3/pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls.md">pt_phi3_microsoft_phi_3_mini_4k_instruct_token_cls</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>146</th>
      <td>qwen</td>
      <td><a href="./models/qwen/pt_qwen1_5_qwen_qwen1_5_0_5b_chat.md">pt_qwen1_5_qwen_qwen1_5_0_5b_chat</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>147</th>
      <td>qwen</td>
      <td><a href="./models/qwen/pt_qwen1_5_qwen_qwen1_5_0_5b_clm.md">pt_qwen1_5_qwen_qwen1_5_0_5b_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>148</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm.md">pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>149</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm.md">pt_qwen_coder_qwen_qwen2_5_coder_1_5b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>150</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm.md">pt_qwen_coder_qwen_qwen2_5_coder_1_5b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>151</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_3b_clm.md">pt_qwen_coder_qwen_qwen2_5_coder_3b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>152</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm.md">pt_qwen_coder_qwen_qwen2_5_coder_3b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>153</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_7b_clm.md">pt_qwen_coder_qwen_qwen2_5_coder_7b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>154</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm.md">pt_qwen_coder_qwen_qwen2_5_coder_7b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>155</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_0_5b_clm.md">pt_qwen_v2_qwen_qwen2_5_0_5b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>156</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm.md">pt_qwen_v2_qwen_qwen2_5_0_5b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>157</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_1_5b_clm.md">pt_qwen_v2_qwen_qwen2_5_1_5b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>158</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm.md">pt_qwen_v2_qwen_qwen2_5_1_5b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>159</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_3b_clm.md">pt_qwen_v2_qwen_qwen2_5_3b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>160</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm.md">pt_qwen_v2_qwen_qwen2_5_3b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>161</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_7b_clm.md">pt_qwen_v2_qwen_qwen2_5_7b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>162</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm.md">pt_qwen_v2_qwen_qwen2_5_7b_instruct_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>163</th>
      <td>rcnn</td>
      <td><a href="./models/rcnn/pt_rcnn_base_rect_0.md">pt_rcnn_base_rect_0</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>164</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_facebook_regnet_y_040.md">pt_regnet_facebook_regnet_y_040</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>165</th>
      <td>regnet</td>
      <td><a href="./models/regnet/pt_regnet_facebook_regnet_y_040_img_cls.md">pt_regnet_facebook_regnet_y_040_img_cls</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>166</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_50_hf.md">pt_resnet_50_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>167</th>
      <td>resnet</td>
      <td><a href="./models/resnet/pt_resnet_50_timm.md">pt_resnet_50_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>168</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext101_32x8d_torchhub.md">pt_resnext_resnext101_32x8d_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>169</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext101_32x8d_wsl_torchhub.md">pt_resnext_resnext101_32x8d_wsl_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>170</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext101_64x4d_osmr.md">pt_resnext_resnext101_64x4d_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>171</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext14_32x4d_osmr.md">pt_resnext_resnext14_32x4d_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>172</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext26_32x4d_osmr.md">pt_resnext_resnext26_32x4d_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>173</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext50_32x4d_osmr.md">pt_resnext_resnext50_32x4d_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>174</th>
      <td>resnext</td>
      <td><a href="./models/resnext/pt_resnext_resnext50_32x4d_torchhub.md">pt_resnext_resnext50_32x4d_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>175</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn101fpn.md">pt_retinanet_retinanet_rn101fpn</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>176</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn152fpn.md">pt_retinanet_retinanet_rn152fpn</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>177</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn18fpn.md">pt_retinanet_retinanet_rn18fpn</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>178</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn34fpn.md">pt_retinanet_retinanet_rn34fpn</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>179</th>
      <td>retinanet</td>
      <td><a href="./models/retinanet/pt_retinanet_retinanet_rn50fpn.md">pt_retinanet_retinanet_rn50fpn</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>180</th>
      <td>roberta</td>
      <td><a href="./models/roberta/pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls.md">pt_roberta_cardiffnlp_twitter_roberta_base_sentiment_seq_cls</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>84 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>181</th>
      <td>roberta</td>
      <td><a href="./models/roberta/pt_roberta_xlm_roberta_base_mlm.md">pt_roberta_xlm_roberta_base_mlm</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>182</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b0_img_cls.md">pt_segformer_nvidia_mit_b0_img_cls</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>183</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b1_img_cls.md">pt_segformer_nvidia_mit_b1_img_cls</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>184</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b2_img_cls.md">pt_segformer_nvidia_mit_b2_img_cls</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>185</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b3_img_cls.md">pt_segformer_nvidia_mit_b3_img_cls</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>186</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b4_img_cls.md">pt_segformer_nvidia_mit_b4_img_cls</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>187</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_mit_b5_img_cls.md">pt_segformer_nvidia_mit_b5_img_cls</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>188</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg.md">pt_segformer_nvidia_segformer_b0_finetuned_ade_512_512_sem_seg</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>189</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg.md">pt_segformer_nvidia_segformer_b1_finetuned_ade_512_512_sem_seg</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>190</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg.md">pt_segformer_nvidia_segformer_b2_finetuned_ade_512_512_sem_seg</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>191</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg.md">pt_segformer_nvidia_segformer_b3_finetuned_ade_512_512_sem_seg</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>192</th>
      <td>segformer</td>
      <td><a href="./models/segformer/pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg.md">pt_segformer_nvidia_segformer_b4_finetuned_ade_512_512_sem_seg</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>193</th>
      <td>squeezebert</td>
      <td><a href="./models/squeezebert/pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls.md">pt_squeezebert_squeezebert_squeezebert_mnli_seq_cls</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>94 %</td>
      <td>90 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>194</th>
      <td>ssd300_resnet50</td>
      <td><a href="./models/ssd300_resnet50/pt_ssd300_resnet50_base.md">pt_ssd300_resnet50_base</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>195</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_large.md">pt_stereo_facebook_musicgen_large</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>196</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_medium.md">pt_stereo_facebook_musicgen_medium</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>197</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_small.md">pt_stereo_facebook_musicgen_small</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>94 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>198</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_google_flan_t5_base_text_gen.md">pt_t5_google_flan_t5_base_text_gen</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>199</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_google_flan_t5_large_text_gen.md">pt_t5_google_flan_t5_large_text_gen</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>200</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_google_flan_t5_small_text_gen.md">pt_t5_google_flan_t5_small_text_gen</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>94 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>201</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_base_text_gen.md">pt_t5_t5_base_text_gen</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>202</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_large_text_gen.md">pt_t5_t5_large_text_gen</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>203</th>
      <td>t5</td>
      <td><a href="./models/t5/pt_t5_t5_small_text_gen.md">pt_t5_t5_small_text_gen</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>94 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>204</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_base.md">pt_unet_base</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>97 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>205</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_cityscape_osmr.md">pt_unet_cityscape_osmr</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>96 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>206</th>
      <td>unet</td>
      <td><a href="./models/unet/pt_unet_qubvel.md">pt_unet_qubvel</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>207</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_19_hf.md">pt_vgg_19_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>208</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_bn_vgg19.md">pt_vgg_bn_vgg19</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>209</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_bn_vgg19b.md">pt_vgg_bn_vgg19b</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>210</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg11.md">pt_vgg_vgg11</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>211</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg13.md">pt_vgg_vgg13</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>212</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg16.md">pt_vgg_vgg16</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>213</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg19.md">pt_vgg_vgg19</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>214</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg19_bn_timm.md">pt_vgg_vgg19_bn_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>215</th>
      <td>vgg</td>
      <td><a href="./models/vgg/pt_vgg_vgg19_bn_torchhub.md">pt_vgg_vgg19_bn_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>216</th>
      <td>vilt</td>
      <td><a href="./models/vilt/pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf.md">pt_vilt_dandelin_vilt_b32_finetuned_vqa_qa_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>217</th>
      <td>vilt</td>
      <td><a href="./models/vilt/pt_vilt_dandelin_vilt_b32_mlm_mlm_hf.md">pt_vilt_dandelin_vilt_b32_mlm_mlm_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>218</th>
      <td>vit</td>
      <td><a href="./models/vit/pt_vit_google_vit_base_patch16_224_img_cls_hf.md">pt_vit_google_vit_base_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>219</th>
      <td>vit</td>
      <td><a href="./models/vit/pt_vit_google_vit_large_patch16_224_img_cls_hf.md">pt_vit_google_vit_large_patch16_224_img_cls_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>220</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_ese_vovnet19b_dw.md">pt_vovnet_ese_vovnet19b_dw</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>221</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_ese_vovnet39b.md">pt_vovnet_ese_vovnet39b</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>222</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_ese_vovnet99b.md">pt_vovnet_ese_vovnet99b</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>223</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_vovnet27s_osmr.md">pt_vovnet_vovnet27s_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>224</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_vovnet39_osmr.md">pt_vovnet_vovnet39_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>225</th>
      <td>vovnet</td>
      <td><a href="./models/vovnet/pt_vovnet_vovnet57_osmr.md">pt_vovnet_vovnet57_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>226</th>
      <td>whisper_0</td>
      <td><a href="./models/whisper_0/pt_whisper_openai_whisper_base.md">pt_whisper_openai_whisper_base</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>227</th>
      <td>whisper_0</td>
      <td><a href="./models/whisper_0/pt_whisper_openai_whisper_large.md">pt_whisper_openai_whisper_large</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>228</th>
      <td>whisper_0</td>
      <td><a href="./models/whisper_0/pt_whisper_openai_whisper_medium.md">pt_whisper_openai_whisper_medium</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>229</th>
      <td>whisper_0</td>
      <td><a href="./models/whisper_0/pt_whisper_openai_whisper_small.md">pt_whisper_openai_whisper_small</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>230</th>
      <td>whisper_0</td>
      <td><a href="./models/whisper_0/pt_whisper_openai_whisper_tiny.md">pt_whisper_openai_whisper_tiny</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>231</th>
      <td>whisper_3</td>
      <td><a href="./models/whisper_3/pt_whisper_openai_whisper_large_v3_turbo_speech_translate.md">pt_whisper_openai_whisper_large_v3_turbo_speech_translate</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>232</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet101_2.md">pt_wideresnet_wide_resnet101_2</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>233</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet101_2_timm.md">pt_wideresnet_wide_resnet101_2_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>234</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet50_2.md">pt_wideresnet_wide_resnet50_2</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>235</th>
      <td>wideresnet</td>
      <td><a href="./models/wideresnet/pt_wideresnet_wide_resnet50_2_timm.md">pt_wideresnet_wide_resnet50_2_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>236</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception41_timm.md">pt_xception_xception41_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>237</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception65_timm.md">pt_xception_xception65_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>85 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>238</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception71_timm.md">pt_xception_xception71_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>239</th>
      <td>xception</td>
      <td><a href="./models/xception/pt_xception_xception_timm.md">pt_xception_xception_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>240</th>
      <td>xglm</td>
      <td><a href="./models/xglm/pt_xglm_facebook_xglm_1_7b_clm.md">pt_xglm_facebook_xglm_1_7b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>241</th>
      <td>xglm</td>
      <td><a href="./models/xglm/pt_xglm_facebook_xglm_564m_clm.md">pt_xglm_facebook_xglm_564m_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>242</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5l_imgcls_torchhub_320x320.md">pt_yolo_v5_yolov5l_imgcls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>243</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5l_imgcls_torchhub_480x480.md">pt_yolo_v5_yolov5l_imgcls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>244</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5l_imgcls_torchhub_640x640.md">pt_yolo_v5_yolov5l_imgcls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>245</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5m_imgcls_torchhub_320x320.md">pt_yolo_v5_yolov5m_imgcls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>246</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5m_imgcls_torchhub_480x480.md">pt_yolo_v5_yolov5m_imgcls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>247</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5m_imgcls_torchhub_640x640.md">pt_yolo_v5_yolov5m_imgcls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>248</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5n_imgcls_torchhub_320x320.md">pt_yolo_v5_yolov5n_imgcls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>249</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5n_imgcls_torchhub_480x480.md">pt_yolo_v5_yolov5n_imgcls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>250</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5n_imgcls_torchhub_640x640.md">pt_yolo_v5_yolov5n_imgcls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>251</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280.md">pt_yolo_v5_yolov5s_imgcls_torchhub_1280x1280</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>96 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>252</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_imgcls_torchhub_320x320.md">pt_yolo_v5_yolov5s_imgcls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>253</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_imgcls_torchhub_480x480.md">pt_yolo_v5_yolov5s_imgcls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>254</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5s_imgcls_torchhub_640x640.md">pt_yolo_v5_yolov5s_imgcls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>255</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5x_imgcls_torchhub_320x320.md">pt_yolo_v5_yolov5x_imgcls_torchhub_320x320</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>256</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5x_imgcls_torchhub_480x480.md">pt_yolo_v5_yolov5x_imgcls_torchhub_480x480</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>257</th>
      <td>yolo_v5</td>
      <td><a href="./models/yolo_v5/pt_yolo_v5_yolov5x_imgcls_torchhub_640x640.md">pt_yolo_v5_yolov5x_imgcls_torchhub_640x640</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>258</th>
      <td>yolo_v6</td>
      <td><a href="./models/yolo_v6/pt_yolo_v6_yolov6l.md">pt_yolo_v6_yolov6l</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>259</th>
      <td>yolo_v6</td>
      <td><a href="./models/yolo_v6/pt_yolo_v6_yolov6m.md">pt_yolo_v6_yolov6m</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>260</th>
      <td>yolo_v6</td>
      <td><a href="./models/yolo_v6/pt_yolo_v6_yolov6n.md">pt_yolo_v6_yolov6n</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>261</th>
      <td>yolo_v6</td>
      <td><a href="./models/yolo_v6/pt_yolo_v6_yolov6s.md">pt_yolo_v6_yolov6s</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>262</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_darknet.md">pt_yolox_yolox_darknet</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>263</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_l.md">pt_yolox_yolox_l</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>264</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_m.md">pt_yolox_yolox_m</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>265</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_nano.md">pt_yolox_yolox_nano</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>266</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_s.md">pt_yolox_yolox_s</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>267</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_tiny.md">pt_yolox_yolox_tiny</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>268</th>
      <td>yolox</td>
      <td><a href="./models/yolox/pt_yolox_yolox_x.md">pt_yolox_yolox_x</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
  </tbody>
</table>
