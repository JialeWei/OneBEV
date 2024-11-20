import onebev
import os
import os.path as osp
import argparse

from mmengine.config import Config
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    modality = osp.dirname(args.config).split('/')[-1]
    experiment_name = osp.splitext(osp.basename(args.config))[0]

    if args.work_dir is not None:
        cfg.work_dir = osp.join(args.work_dir, modality, experiment_name)
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', modality, experiment_name)

    runner = Runner.from_cfg(cfg)

    os.system(
        f'rm -rf {os.path.join(runner.work_dir, f"{experiment_name}.py")}'
    )  
    cfg.dump(
        os.path.join(runner.log_dir,
                     f'{experiment_name}.py'))
    runner.train()

if __name__ == '__main__':
    main()
