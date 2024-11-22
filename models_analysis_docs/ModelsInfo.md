<h1>List of models and current compiler support rates</h1>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th colspan="3" halign="left">Model Details</th>
      <th colspan="4" halign="left">Passing rate of unique ops for each component</th>
    </tr>
    <tr>
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
      <td>albert</td>
      <td><a href="./Models/albert/pt_albert_xxlarge_v2_masked_lm.md">pt_albert_xxlarge_v2_masked_lm</a></td>
      <td>pytorch</td>
      <td>78 %</td>
      <td>78 %</td>
      <td>71 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <td>albert</td>
      <td><a href="./Models/albert/pt_albert_large_v2_token_cls.md">pt_albert_large_v2_token_cls</a></td>
      <td>pytorch</td>
      <td>86 %</td>
      <td>86 %</td>
      <td>78 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <td>bart</td>
      <td><a href="./Models/bart/pt_bart.md">pt_bart</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>70 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>bert</td>
      <td><a href="./Models/bert/pt_bert_masked_lm.md">pt_bert_masked_lm</a></td>
      <td>pytorch</td>
      <td>82 %</td>
      <td>82 %</td>
      <td>73 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <td>codegen</td>
      <td><a href="./Models/codegen/pt_codegen_350M_mono.md">pt_codegen_350M_mono</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>74 %</td>
      <td>7 %</td>
    </tr>
    <tr>
      <td>distilbert</td>
      <td><a href="./Models/distilbert/pt_distilbert_question_answering.md">pt_distilbert_question_answering</a></td>
      <td>pytorch</td>
      <td>80 %</td>
      <td>80 %</td>
      <td>72 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <td>distilbert</td>
      <td><a href="./Models/distilbert/pt_distilbert_sequence_classification.md">pt_distilbert_sequence_classification</a></td>
      <td>pytorch</td>
      <td>84 %</td>
      <td>84 %</td>
      <td>74 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <td>distilbert</td>
      <td><a href="./Models/distilbert/pt_distilbert_masked_lm.md">pt_distilbert_masked_lm</a></td>
      <td>pytorch</td>
      <td>81 %</td>
      <td>81 %</td>
      <td>72 %</td>
      <td>6 %</td>
    </tr>
    <tr>
      <td>distilbert</td>
      <td><a href="./Models/distilbert/pt_distilbert_token_classification.md">pt_distilbert_token_classification</a></td>
      <td>pytorch</td>
      <td>82 %</td>
      <td>82 %</td>
      <td>73 %</td>
      <td>6 %</td>
    </tr>
    <tr>
      <td>dpr</td>
      <td><a href="./Models/dpr/pt_dpr_reader_multiset_base.md">pt_dpr_reader_multiset_base</a></td>
      <td>pytorch</td>
      <td>80 %</td>
      <td>80 %</td>
      <td>67 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <td>dpr</td>
      <td><a href="./Models/dpr/pt_dpr_ctx_encoder_multiset_base.md">pt_dpr_ctx_encoder_multiset_base</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>71 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <td>dpr</td>
      <td><a href="./Models/dpr/pt_dpr_question_encoder_multiset_base.md">pt_dpr_question_encoder_multiset_base</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>71 %</td>
      <td>5 %</td>
    </tr>
    <tr>
      <td>gpt2</td>
      <td><a href="./Models/gpt2/pt_gpt2_generation.md">pt_gpt2_generation</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>87 %</td>
      <td>68 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <td>gptneo</td>
      <td><a href="./Models/gptneo/pt_gpt_neo_125M_causal_lm.md">pt_gpt_neo_125M_causal_lm</a></td>
      <td>pytorch</td>
      <td>85 %</td>
      <td>85 %</td>
      <td>65 %</td>
      <td>6 %</td>
    </tr>
    <tr>
      <td>llama3</td>
      <td><a href="./Models/llama3/pt_Meta_Llama_3_8B_seq_cls.md">pt_Meta_Llama_3_8B_seq_cls</a></td>
      <td>pytorch</td>
      <td>82 %</td>
      <td>80 %</td>
      <td>65 %</td>
      <td>10 %</td>
    </tr>
    <tr>
      <td>llama3</td>
      <td><a href="./Models/llama3/pt_Meta_Llama_3_8B_Instruct_seq_cls.md">pt_Meta_Llama_3_8B_Instruct_seq_cls</a></td>
      <td>pytorch</td>
      <td>82 %</td>
      <td>80 %</td>
      <td>65 %</td>
      <td>10 %</td>
    </tr>
    <tr>
      <td>opt</td>
      <td><a href="./Models/opt/pt_opt_125m_causal_lm.md">pt_opt_125m_causal_lm</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>72 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <td>phi2</td>
      <td><a href="./Models/phi2/pt_phi_2_causal_lm.md">pt_phi_2_causal_lm</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>67 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <td>phi3</td>
      <td><a href="./Models/phi3/pt_phi_3_mini_4k_instruct_causal_lm.md">pt_phi_3_mini_4k_instruct_causal_lm</a></td>
      <td>pytorch</td>
      <td>88 %</td>
      <td>86 %</td>
      <td>72 %</td>
      <td>8 %</td>
    </tr>
    <tr>
      <td>qwen</td>
      <td><a href="./Models/qwen/pt_qwen_causal_lm.md">pt_qwen_causal_lm</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>94 %</td>
      <td>76 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>roberta</td>
      <td><a href="./Models/roberta/pt_roberta_masked_lm.md">pt_roberta_masked_lm</a></td>
      <td>pytorch</td>
      <td>84 %</td>
      <td>84 %</td>
      <td>71 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <td>squeezebert</td>
      <td><a href="./Models/squeezebert/pt_squeezebert.md">pt_squeezebert</a></td>
      <td>pytorch</td>
      <td>87 %</td>
      <td>84 %</td>
      <td>72 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <td>xglm</td>
      <td><a href="./Models/xglm/pt_xglm_564M.md">pt_xglm_564M</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>72 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <td>alexnet</td>
      <td><a href="./Models/alexnet/pt_alexnet_torchhub.md">pt_alexnet_torchhub</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>75 %</td>
      <td>59 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>autoencoder</td>
      <td><a href="./Models/autoencoder/pt_linear_ae.md">pt_linear_ae</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>autoencoder</td>
      <td><a href="./Models/autoencoder/pt_conv_ae.md">pt_conv_ae</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>64 %</td>
      <td>53 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>clip</td>
      <td><a href="./Models/clip/pt_clip_text_model.md">pt_clip_text_model</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>91 %</td>
      <td>70 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <td>deit</td>
      <td><a href="./Models/deit/pt_deit_base_distilled_patch16_224.md">pt_deit_base_distilled_patch16_224</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>92 %</td>
      <td>72 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>dla</td>
      <td><a href="./Models/dla/pt_dla169.md">pt_dla169</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>89 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>efficientnet</td>
      <td><a href="./Models/efficientnet/pt_efficientnet_b4_timm.md">pt_efficientnet_b4_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>85 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>ghostnet</td>
      <td><a href="./Models/ghostnet/pt_ghostnet_100.md">pt_ghostnet_100</a></td>
      <td>pytorch</td>
      <td>97 %</td>
      <td>91 %</td>
      <td>81 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>googlenet</td>
      <td><a href="./Models/googlenet/pt_googlenet.md">pt_googlenet</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>94 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>inception_v4</td>
      <td><a href="./Models/inception_v4/pt_timm_inception_v4.md">pt_timm_inception_v4</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>90 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>mlp_mixer</td>
      <td><a href="./Models/mlp_mixer/pt_mixer_s32_224.md">pt_mixer_s32_224</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>88 %</td>
      <td>82 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>mobilenet_v1</td>
      <td><a href="./Models/mobilenet_v1/pt_mobilenet_v1_basic.md">pt_mobilenet_v1_basic</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>81 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>mobilenet_v1</td>
      <td><a href="./Models/mobilenet_v1/pt_mobilenet_v1_224.md">pt_mobilenet_v1_224</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>87 %</td>
      <td>79 %</td>
      <td>4 %</td>
    </tr>
    <tr>
      <td>mobilenet_v2</td>
      <td><a href="./Models/mobilenet_v2/mobilenetv2_deeplabv3.md">mobilenetv2_deeplabv3</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>86 %</td>
      <td>77 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <td>mobilenet_v3</td>
      <td><a href="./Models/mobilenet_v3/pt_mobilenet_v3_large.md">pt_mobilenet_v3_large</a></td>
      <td>pytorch</td>
      <td>91 %</td>
      <td>85 %</td>
      <td>74 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>monodle</td>
      <td><a href="./Models/monodle/pt_monodle.md">pt_monodle</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>92 %</td>
      <td>79 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>nbeats</td>
      <td><a href="./Models/nbeats/nbeats_seasonality.md">nbeats_seasonality</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>94 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>nbeats</td>
      <td><a href="./Models/nbeats/nbeats_trend.md">nbeats_trend</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>nbeats</td>
      <td><a href="./Models/nbeats/nbeats_generic.md">nbeats_generic</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>91 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>perceiverio</td>
      <td><a href="./Models/perceiverio/pt_vision_perceiver_fourier.md">pt_vision_perceiver_fourier</a></td>
      <td>pytorch</td>
      <td>92 %</td>
      <td>92 %</td>
      <td>79 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnet</td>
      <td><a href="./Models/resnet/pt_resnet50_timm.md">pt_resnet50_timm</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnet</td>
      <td><a href="./Models/resnet/pt_resnet50.md">pt_resnet50</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnext</td>
      <td><a href="./Models/resnext/pt_resnext50_torchhub.md">pt_resnext50_torchhub</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnext</td>
      <td><a href="./Models/resnext/pt_resnext101_osmr.md">pt_resnext101_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnext</td>
      <td><a href="./Models/resnext/pt_resnext14_osmr.md">pt_resnext14_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>95 %</td>
      <td>82 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnext</td>
      <td><a href="./Models/resnext/pt_resnext101_torchhub.md">pt_resnext101_torchhub</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>94 %</td>
      <td>87 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnext</td>
      <td><a href="./Models/resnext/pt_resnext101_fb_wsl.md">pt_resnext101_fb_wsl</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>99 %</td>
      <td>92 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnext</td>
      <td><a href="./Models/resnext/pt_resnext50_osmr.md">pt_resnext50_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>97 %</td>
      <td>86 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>resnext</td>
      <td><a href="./Models/resnext/pt_resnext26_osmr.md">pt_resnext26_osmr</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>96 %</td>
      <td>81 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>retinanet</td>
      <td><a href="./Models/retinanet/pt_retinanet_rn152fpn.md">pt_retinanet_rn152fpn</a></td>
      <td>pytorch</td>
      <td>96 %</td>
      <td>93 %</td>
      <td>86 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <td>segformer_semseg</td>
      <td><a href="./Models/segformer_semseg/pt_segformer_b1_finetuned_ade_512_512.md">pt_segformer_b1_finetuned_ade_512_512</a></td>
      <td>pytorch</td>
      <td>93 %</td>
      <td>90 %</td>
      <td>77 %</td>
      <td>2 %</td>
    </tr>
    <tr>
      <td>swin</td>
      <td><a href="./Models/swin/pt_swin_tiny_patch4_window7_224.md">pt_swin_tiny_patch4_window7_224</a></td>
      <td>pytorch</td>
      <td>78 %</td>
      <td>77 %</td>
      <td>56 %</td>
      <td>16 %</td>
    </tr>
    <tr>
      <td>vilt</td>
      <td><a href="./Models/vilt/pt_ViLt_maskedlm.md">pt_ViLt_maskedlm</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>73 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>vilt</td>
      <td><a href="./Models/vilt/pt_ViLt_question_answering.md">pt_ViLt_question_answering</a></td>
      <td>pytorch</td>
      <td>88 %</td>
      <td>88 %</td>
      <td>72 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>vit</td>
      <td><a href="./Models/vit/pt_vit_large_patch16_224.md">pt_vit_large_patch16_224</a></td>
      <td>pytorch</td>
      <td>94 %</td>
      <td>92 %</td>
      <td>72 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>xception</td>
      <td><a href="./Models/xception/pt_xception71_timm.md">pt_xception71_timm</a></td>
      <td>pytorch</td>
      <td>86 %</td>
      <td>84 %</td>
      <td>76 %</td>
      <td>0 %</td>
    </tr>
  </tbody>
</table>
