# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import sys
import os
import torch


class PyBudify(torch.nn.Module):
    def __init__(self, pt_module, device='silicon', arch='wormhole_b0', precision='fp32', amp_level=0, micro_batch_size=1, fuse=False, num_chips=1, perf=None, verify=False, log_level='ERROR', tti_save=None, tti_load=None):
        super().__init__()

        self.device = device
        self.bound_module = pt_module
        self.tti_save = tti_save
        self.tti_load = tti_load

        if device != 'pytorch':
            # forge workarounds
            os.environ["GOLDEN_WORMHOLE_B0"] = "1"
            os.environ["FORGE_ENABLE_BROADCAST_SPLITTING"] = "1"
            #os.environ["FORGE_DISABLE_FORK_JOIN_BUF"] = "1"
            os.environ["FORGE_DRAM_PICK_CAPACITY"] = "1"
            os.environ["WHA0_DISABLE_RELAY_BUFS"] = "1"
            #os.environ["FORGE_FUSE_STOP_ON_RECIPROCAL"] = "1"
            os.environ["FORGE_PLACER_SNAKE"] = "1"
            os.environ["LOGGER_LEVEL"] = log_level
            os.environ["LOGURU_LEVEL"] = log_level

            forge = self.forge = __import__('forge') # let us set log levels before importing forge

        if device == 'pytorch':
            pass
        else:
            devtype = { 'golden' : forge.BackendType.Golden,
                        'silicon': forge.BackendType.Silicon,
                      }[device]

            module = forge.PyTorchModule("pybudify_module", self.bound_module)

            if precision == 'fp32':
                fallback = forge.DataFormat.Float32
            elif precision == 'fp16':
                fallback = forge.DataFormat.Float16
            elif precision == 'bf16':
                fallback = forge.DataFormat.Float16_b
            elif precision == 'fp8':
                fallback = forge.DataFormat.Bfp8
            elif precision == 'fp8b':
                fallback = forge.DataFormat.Bfp8_b
            else:
                raise ValueError('Precision "%s" not implemented' % precision)

#            if manual_placement:
#                manual_placer(forge.config, manual_placement)

            OFFSET = 65 - 7 # = 58
            for layer_num in range(24):
                k = OFFSET * layer_num
                #forge.config.set_epoch_break([f'add_{17+k}', f'matmul_{16+k}'])
                forge.config.add_schedule_constraint([f'pybudify_module.output_transpose_{9+k}_tm_nop', f'add_{14+k}_output_nop_0', f'concatenate_{35+k}.dc.concatenate.2'])

            perf_level = { None    : None,
                          'none'   : None,
                          'light'  : forge.PerfTraceLevel.LIGHT,
                          'verbose': forge.PerfTraceLevel.VERBOSE }[perf]
            forge.set_configuration_options(default_df_override=fallback, accumulate_df=fallback, amp_level=amp_level, enable_auto_fusing=fuse, performance_trace=perf_level, backend_opt_level=3)

            forge_arch = { 'grayskull': forge.BackendDevice.Grayskull,
                            'wormhole_b0': forge.BackendDevice.Wormhole_B0 }[arch]
            
            if tti_load is not None:
                self.tt0 = forge.TTDevice.load_image(img_path=tti_load)
            else:
                self.tt0 = forge.TTDevice('tt0', module=module,
                                            fp32_fallback=fallback,
                                            arch=forge_arch,
                                            devtype=devtype,
                                            chip_ids=list(range(num_chips)))
                    
            mp = torch.multiprocessing.get_context('spawn')
            self.output_q = mp.Queue()

            if verify:
                self.verify_cfg = forge.VerifyConfig(verify_all=True,
                                                      verify_last=True,
                                                      devtype=forge.BackendType.Silicon,
                                                      arch=forge_arch,)
            else:
                self.verify_cfg = None

            self.initialized = False
            self.micro_batch_size = micro_batch_size


    def __call__(self, *args, **kwargs):
        if self.device == 'pytorch':
            result = self.bound_module(*args, **kwargs)
        else:
            if not self.initialized:
                if self.tti_save is not None:
                    self.tt0.compile_to_image(
                        img_path=self.tti_save,
                        training=False,
                        sample_inputs=args,
                        microbatch_count=self.micro_batch_size,
                    )
                    print(f'Saved image to {self.tti_save}')
                    sys.exit(0)
                self.forge.initialize_pipeline(training=False,
                                        sample_inputs=args,
                                        output_queue=self.output_q,
                                        microbatch_count=self.micro_batch_size,
                                        _sequential=True, # FIXME: can we implement concurrent mode and still have a wrapper?
                                        _verify_cfg=self.verify_cfg,
                                        )
                self.initialized = True

            self.tt0.push_to_inputs(*args)
            self.forge.run_forward(input_count=1, _sequential=True)
            ys = self.output_q.get()
            outputs = tuple([ y.value().float() for y in ys if isinstance(y, self.forge.tensor.TensorFromPytorch)])
            if len(outputs) == 1:
                outputs = outputs[0]
            if self.verify_cfg:
                baseline = self.bound_module(*args, **kwargs)
                if len(outputs) != len(baseline):
                    print(f'Num outputs: {len(outputs)}, expected: {len(baseline)}')
                for i, (real, expected) in enumerate(zip(outputs, baseline)):
                    pcc = torch.corrcoef(torch.stack([real.view(-1), expected.view(-1)]))[0,1]
                    print('PCC tensor %d: %.3f' % (i, pcc))

            result = outputs

        return result

