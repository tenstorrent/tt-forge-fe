# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import json
import os

def apply_mlp(forge, config):
    # Config could be df string or a dict of df strings
    if isinstance(config, str):
        df = str_to_dataformat(forge, config)
        gate_df = up_df = down_df = df
    else:
        gate_df = str_to_dataformat(forge, config['gate'])
        up_df = str_to_dataformat(forge, config['up'])
        down_df = str_to_dataformat(forge, config['down'])

    # MLP dataformat is applied to MLP weights with this regex
    forge.config.configure_mixed_precision(
        output_df=gate_df,
        name_regex=".*mlp.gate_proj.weight.*",
        input_df={0: [gate_df, True]})

    forge.config.configure_mixed_precision(
        output_df=up_df,
        name_regex=".*mlp.up_proj.weight.*",
        input_df={0: [up_df, True]})

    forge.config.configure_mixed_precision(
        output_df=down_df,
        name_regex=".*mlp.down_proj.weight.*",
        input_df={0: [down_df, True]})


def apply_attn(forge, config):
    # Config could be df string or a dict of df strings
    if isinstance(config, str):
        df = str_to_dataformat(forge, config)
        q_df = k_df = v_df = o_df = df
    else:
        q_df = str_to_dataformat(forge, config['q'])
        k_df = str_to_dataformat(forge, config['k'])
        v_df = str_to_dataformat(forge, config['v'])
        o_df = str_to_dataformat(forge, config['o'])

    # Attention dataformat is applied to attention weights with this regex
    forge.config.configure_mixed_precision(
        output_df=q_df,
        name_regex=".*self_attn.q_proj.weight.*",
        input_df={0: [q_df, True]})
    forge.config.configure_mixed_precision(
        output_df=k_df,
        name_regex=".*self_attn.k_proj.weight.*",
        input_df={0: [k_df, True]})
    forge.config.configure_mixed_precision(
        output_df=v_df,
        name_regex=".*self_attn.v_proj.weight.*",
        input_df={0: [v_df, True]})
    forge.config.configure_mixed_precision(
        output_df=o_df,
        name_regex=".*self_attn.o_proj.weight.*",
        input_df={0: [o_df, True]})


def apply_cache(forge, config, num_layers):
    # Config could be df string or a dict of df strings
    if isinstance(config, str):
        df = str_to_dataformat(forge, config)
        key_df = df
        value_df = df
    else:
        key_df = str_to_dataformat(forge, config['key'])
        value_df = str_to_dataformat(forge, config['value'])

    forge.config.configure_mixed_precision(
        output_df=key_df,
        name_regex="k_past_.*",
        input_df={0: [key_df, True]})

    forge.config.configure_mixed_precision(
        output_df=value_df,
        name_regex="v_past_.*",
        input_df={0: [value_df, True]})
    

    # Also let's loop over the concatenate ops and make sure they are using this DF.
    # Otherwise we get garbage outputs. Bad.
    # TODO: Figure out a more programmatic way to figure out these op names
    OP_OFFSET = 77
    INDEX_START = num_layers * OP_OFFSET
    HSTACK_OFFSET = 4
    for i in range(num_layers):
        k = OP_OFFSET * i
        j = HSTACK_OFFSET * i
        # special-case key ops
        forge.config.configure_mixed_precision(
            output_df=key_df,
            name_regex=f'concatenate_{30+k}.dc.concatenate.0',
            input_df={0: [key_df, True], 1: [key_df, True]})

        # Write-view also needs overriding
        forge.config.configure_mixed_precision(
            output_df=key_df,
            name_regex=f".*output_hstack_{INDEX_START + 1 +j}.*",
            input_df={0: [key_df, True]})

        # special-case value ops
        forge.config.configure_mixed_precision(
            output_df=value_df,
            name_regex=f'concatenate_{44+k}.dc.concatenate.0',
            input_df={0: [value_df, True], 1: [value_df, True]})

        # Write-view also needs overriding
        forge.config.configure_mixed_precision(
            output_df=value_df,
            name_regex=f".*output_hstack_{INDEX_START + 3 +j}.*",
            input_df={0: [value_df, True]})


def apply_matmul_acc(forge, df):
    forge.config.configure_mixed_precision(
        op_type="matmul",
        intermediate_df=df,
        accumulate_df=df,
    )


def apply_default(forge, df):
    # Default dataformat is applied to all other weights with this regex
    forge.set_configuration_options(default_df_override=df, accumulate_df=df)


def apply_attn_mask(forge, df):
    # MLP dataformat is applied to MLP weights with this regex
    forge.config.configure_mixed_precision(
    output_df=df,
    name_regex="attention_mask",
    input_df={0: [df, True]})


def str_to_dataformat(forge, df_str):
    if df_str == 'fp32':
        df = forge.DataFormat.Float32
    elif df_str == 'fp16':
        df = forge.DataFormat.Float16
    elif df_str == 'bf16':
        df = forge.DataFormat.Float16_b
    elif df_str == 'fp8':
        df = forge.DataFormat.Bfp8
    elif df_str == 'fp8b':
        df = forge.DataFormat.Bfp8_b
    elif df_str == 'fp4b':
        df = forge.DataFormat.Bfp4_b
    elif df_str == 'fp2b':
        df = forge.DataFormat.Bfp2_b
    else:
        raise ValueError('Precision "%s" not implemented' % precision)
    return df

def apply_amp_settings(forge, config_file, num_layers):
    print('Applying AMP from file ', config_file, flush=True)
    # Open config json
    with open(config_file) as f:
        config = json.load(f)

    '''
    For now, this file has hard-coded ideas of what AMP means, as it applies to Llama.
    Ex: MLP amp means set MLP weights to some df.
    '''
    for k, v in config.items():
        if k == "mm_acc_df":
            apply_matmul_acc(forge, str_to_dataformat(forge, v))
        elif k == "mlp_df":
            apply_mlp(forge, v)
        elif k == "attn_df":
            apply_attn(forge, v)
        elif k == "cache_df":
            apply_cache(forge, v, num_layers)
        elif k == "default_df":
            apply_default(forge, str_to_dataformat(forge, v))
        elif k == "attn_mask_df":
            apply_attn_mask(forge, str_to_dataformat(forge, v))
        else:
            raise ValueError('Config "%s" not implemented' % k)

