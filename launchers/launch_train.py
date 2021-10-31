import time
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from Pointnet_Pointnet2_pytorch.train_semseg_haptic import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    # exp_prefix = '1017-pn-seuss-lr-schedule-larger-pos-weight'
    # exp_prefix = '1025-pn-force-non-shared-visual'
    exp_prefix = '1030-pn-force-non-shared-overfit-to-1'
    exp_prefix = '1030-pn-force-non-shared-better-visual'
    vg = VariantGenerator()

    if not debug:
        vg.add('use_batch_norm', [True, False])
        vg.add('correct_z_rotation', [2])
        vg.add('epoch', [300])
        vg.add('loss_pos_weight', [-1])
        vg.add('batch_size', [1])
        vg.add('npoint', [-1])
        vg.add('seed', [100])
        vg.add('manual_lr_adjust', [False])
        vg.add('schedule_lr', [True])
        vg.add('data_dir', ['2021-10-21-balance'])
        # vg.add('loss_on_contact', [True, False])
        vg.add('force_loss_mode', ['balance'])
        vg.add('normal_loss_mode', ['balance'])
        vg.add('force_loss_weight', [1])
        vg.add('normal_loss_weight', [1])
        vg.add('plot_interval', [20])
        vg.add('separate_model', [True])
        # vg.add('load_dir', ['./data/seuss/1023-pn-force-non-shared/1023-pn-force-non-shared_2021_10_23_23_59_23_0002'])
        vg.add('load_dir', [None])
        vg.add('train', [True])
    else:
        vg.add('correct_z_rotation', [2])
        vg.add('use_batch_norm', [True])
        vg.add('epoch', [100])
        vg.add('plot_interval', [100])
        vg.add('seed', [100])
        vg.add('loss_pos_weight', [-1])
        vg.add('batch_size', [1])
        vg.add('npoint', [-1])
        vg.add('manual_lr_adjust', [False])
        vg.add('schedule_lr', [True])
        vg.add('data_dir', ['2021-10-21'])
        vg.add('loss_on_contact', [True])
        vg.add('force_loss_weight', [1])
        vg.add('normal_loss_weight', [1])
        vg.add('separate_model', [True])
        vg.add('load_dir', ['./data/seuss/1023-pn-force-non-shared/1023-pn-force-non-shared_2021_10_23_23_59_23_0002'])
        vg.add('train', [False])

        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss', 'autobot']:
            if idx == 0:
                # compile_script = 'compile.sh'  # For the first experiment, compile the current softgym
                compile_script = None  # For the first experiment, compile the current softgym
                wait_compile = None
            else:
                compile_script = None
                wait_compile = None  # Wait 30 seconds for the compilation to finish
        elif mode == 'ec2':
            compile_script = 'compile_1.0.sh'
            wait_compile = None
        else:
            compile_script = wait_compile = None
        if hostname.startswith('autobot') and gpu_num > 0:
            env_var = {'CUDA_VISIBLE_DEVICES': str(idx % gpu_num)}
        else:
            env_var = None
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
