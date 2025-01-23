<h1>List of models and current compiler support rates</h1>
<p><b>Last updated date and time(in GMT) :</b> Wednesday, 22 Jan 2025 04:38:32 PM</p><p><b>Commit Id :</b> <a href="https://github.com/tenstorrent/tt-forge-fe/commit/c2d21b329cff469e871f01c94e4d8c1fdefbd4b8">c2d21b329cff469e871f01c94e4d8c1fdefbd4b8</a></p><p><b>Note:</b> For detailed insights into compiler failures and their effects on models, please refer to the <a href="./stats/compiler_analysis_report.md">compiler_analysis_report.md</a>.</p><table border="1" class="dataframe">
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
      <td><a href="./models/albert/pt_albert_base_v2_mlm.md">pt_albert_base_v2_mlm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bart</td>
      <td><a href="./models/bart/pt_bart_facebook_bart_large_mnli_seq_cls.md">pt_bart_facebook_bart_large_mnli_seq_cls</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_mono_clm.md">pt_codegen_salesforce_codegen_350m_mono_clm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>5</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_multi_clm.md">pt_codegen_salesforce_codegen_350m_multi_clm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>6</th>
      <td>codegen</td>
      <td><a href="./models/codegen/pt_codegen_salesforce_codegen_350m_nl_clm.md">pt_codegen_salesforce_codegen_350m_nl_clm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder.md">pt_dpr_facebook_dpr_ctx_encoder_multiset_base_context_encoder</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dpr</td>
      <td><a href="./models/dpr/pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder.md">pt_dpr_facebook_dpr_ctx_encoder_single_nq_base_context_encoder</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>9</th>
      <td>gpt2</td>
      <td><a href="./models/gpt2/pt_gpt2_gpt2_text_gen.md">pt_gpt2_gpt2_text_gen</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>88 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>10</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_125m_clm.md">pt_gptneo_eleutherai_gpt_neo_125m_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>11</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_1_3b_clm.md">pt_gptneo_eleutherai_gpt_neo_1_3b_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>12</th>
      <td>gptneo</td>
      <td><a href="./models/gptneo/pt_gptneo_eleutherai_gpt_neo_2_7b_clm.md">pt_gptneo_eleutherai_gpt_neo_2_7b_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>89 %</td>
      <td>84 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>13</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_125m_clm.md">pt_opt_facebook_opt_125m_clm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>14</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_1_3b_clm.md">pt_opt_facebook_opt_1_3b_clm</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>15</th>
      <td>opt</td>
      <td><a href="./models/opt/pt_opt_facebook_opt_350m_clm.md">pt_opt_facebook_opt_350m_clm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>88 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>16</th>
      <td>qwen_coder</td>
      <td><a href="./models/qwen_coder/pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm.md">pt_qwen_coder_qwen_qwen2_5_coder_0_5b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>17</th>
      <td>qwen_v2</td>
      <td><a href="./models/qwen_v2/pt_qwen_v2_qwen_qwen2_5_0_5b_clm.md">pt_qwen_v2_qwen_qwen2_5_0_5b_clm</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>93 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>18</th>
      <td>roberta</td>
      <td><a href="./models/roberta/pt_roberta_xlm_roberta_base_mlm.md">pt_roberta_xlm_roberta_base_mlm</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>19</th>
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
