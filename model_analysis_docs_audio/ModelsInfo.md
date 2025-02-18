<h1>List of models and current compiler support rates</h1>
<p><b>Last updated date and time(in GMT) :</b> Tuesday, 18 Feb 2025 03:21:00 PM</p><p><b>Commit Id :</b> <a href="https://github.com/tenstorrent/tt-forge-fe/commit/d8fed5549cf3de1018c56bc38f48de329cb8b27c">d8fed5549cf3de1018c56bc38f48de329cb8b27c</a></p><p><b>Note:</b> For detailed insights into compiler failures and their effects on models, please refer to the <a href="./stats/compiler_analysis_report.md">compiler_analysis_report.md</a>.</p><table border="1" class="dataframe">
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
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_large_music_generation_hf.md">pt_stereo_facebook_musicgen_large_music_generation_hf</a></td>
      <td>pytorch</td>
      <td>89 %</td>
      <td>89 %</td>
      <td>82 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_medium_music_generation_hf.md">pt_stereo_facebook_musicgen_medium_music_generation_hf</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>83 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stereo</td>
      <td><a href="./models/stereo/pt_stereo_facebook_musicgen_small_music_generation_hf.md">pt_stereo_facebook_musicgen_small_music_generation_hf</a></td>
      <td>pytorch</td>
      <td>90 %</td>
      <td>90 %</td>
      <td>83 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_base_speech_recognition_hf.md">pt_whisper_openai_whisper_base_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>5</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_large_speech_recognition_hf.md">pt_whisper_openai_whisper_large_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>6</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_medium_speech_recognition_hf.md">pt_whisper_openai_whisper_medium_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>98 %</td>
      <td>98 %</td>
      <td>97 %</td>
      <td>1 %</td>
    </tr>
    <tr>
      <th>7</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_small_speech_recognition_hf.md">pt_whisper_openai_whisper_small_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>98 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>8</th>
      <td>whisper</td>
      <td><a href="./models/whisper/pt_whisper_openai_whisper_tiny_speech_recognition_hf.md">pt_whisper_openai_whisper_tiny_speech_recognition_hf</a></td>
      <td>pytorch</td>
      <td>99 %</td>
      <td>99 %</td>
      <td>97 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <th>9</th>
      <td>whisper_large_v3</td>
      <td><a href="./models/whisper_large_v3/pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf.md">pt_whisper_openai_whisper_large_v3_turbo_speech_translate_hf</a></td>
      <td>pytorch</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>100 %</td>
      <td>0 %</td>
    </tr>
  </tbody>
</table>
