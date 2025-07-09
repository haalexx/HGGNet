import os
import argparse
from utils.general import (check_requirements, check_file, check_yaml, get_latest_run, increment_path,
                           create_experiment_dir)
from pathlib import Path


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Completion')
    parser.add_argument('--name', type=str, default='HGGNet_20250626', help='Name of the experiment')
    parser.add_argument('--project', default='./experiments', help='save to project dir')
    parser.add_argument('--data', type=str, default='./cfgs/dataset_configs/PCN.yaml',
                        help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='./cfgs/model_configs/HGGNet_7_4.yaml',
                        help='model.yaml path')
    parser.add_argument('--distributed', type=bool, default=False, help='job launcher')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--resume', nargs='?', default=False, help='resume most recent training')
    parser.add_argument('--device', default='1', help='cuda_device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--test', type=bool, default=False, help='evaluate the model')
    parser.add_argument('--inference', type=bool, default=False, help='inference the point cloud')
    parser.add_argument('--weights', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--multi_process', action='store_true', default=True)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.resume:  # resume an interrupted run
        # args.experiment_path = args.resume if isinstance(args.resume, str) else get_latest_run(search_dir='./experiments')
        args.experiment_path = os.path.join(args.project, args.name)
        # specified or most recent path
        ckpt = os.path.join(args.experiment_path, 'ckpt-last.pth')
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        args.resume_weights, args.resume = ckpt, True  # reinstate
        args.tfboard_path = os.path.join(args.experiment_path, 'TFBoard')
        # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
        #                 filename=os.path.join(args.experiment_path, 'train.log'), level=logging.INFO)
        # logging.info('======================================================')
        # logging.info(f'Resuming training from {ckpt}')
    else:
        args.data, args.cfg, args.weights, args.project = \
            check_file(args.data), check_yaml(args.cfg), str(args.weights), str(args.project)  # checks
        assert len(args.cfg) or len(args.weights), 'either --cfg or --weights must be specified'
        args.experiment_path = os.path.join(args.project, args.name)
        args.tfboard_path = os.path.join(args.experiment_path, 'TFBoard')
        args.log_name = args.name
        create_experiment_dir(args)

    return args
