<h1>List of models and current compiler support rates</h1>
<p><b>Last updated date and time(in GMT) :</b> Tuesday, 28 Jan 2025 08:35:31 AM</p><p><b>Commit Id :</b> <a href="https://github.com/tenstorrent/tt-forge-fe/commit/11b7d2cd07e33d005a31a6b48b24d6a8e1d1728b">11b7d2cd07e33d005a31a6b48b24d6a8e1d1728b</a></p><p><b>Note:</b> For detailed insights into compiler failures and their effects on models, please refer to the <a href="./stats/compiler_analysis_report.md">compiler_analysis_report.md</a>.</p><table border="1" class="dataframe">
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
      <td>bart</td>
      <td><a href="./models/bart/pt_bart_facebook_bart_large_mnli_seq_cls.md">pt_bart_facebook_bart_large_mnli_seq_cls</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_mono_clm.md">pt_codegen_salesforce_codegen_350m_mono_clm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder.md">pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>5</th>
      <td>gpt2</td>
      <td><a href="./models/gpt2/pt_gpt2_gpt2_text_gen.md">pt_gpt2_gpt2_text_gen</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>88 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>6</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_125m_clm.md">pt_gptneo_eleutherai_gpt_neo_125m_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>7</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_clm.md">pt_opt_facebook_opt_125m_clm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>8</th>
      <td>qwen</td>
      <td><a href="./models/qwen/pt_qwen1_5_qwen_qwen1_5_0_5b_clm.md">pt_qwen1_5_qwen_qwen1_5_0_5b_clm</a></td>
      <td>pytorch</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>95 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>9</th>
      <td>roberta</td>
      <td><a href="./models/roberta/pt_roberta_xlm_roberta_base_mlm.md">pt_roberta_xlm_roberta_base_mlm</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>10</th>
      <td>xglm</td>
      <td><a href="./models/xglm/pt_xglm_facebook_xglm_564m_clm.md">pt_xglm_facebook_xglm_564m_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
  </tbody>
</table>
