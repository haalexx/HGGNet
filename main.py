import sys
import torch
import utils.misc as misc
import time
from tensorboardX import SummaryWriter
from utils import parser, dist_utils
from utils.config import *
from utils.logger import *
from utils.torch_utils import select_device, print_device
from core import train
from core import test

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
# RANK = int(os.getenv('RANK', -1))


def main():
    # get args
    args = parser.get_args()

    # config
    data_config, model_config = get_config(args)

    # CUDA
    select_device(args.device)
    
    # init distributed env first, since logger depends on the dist info.
    if args.distributed:
        # DDP mode
        dist_utils.init_dist('pytorch')
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.name)
    
    print_log(f"'Distributed training:' {args.distributed}", logger=logger)
    if args.local_rank in [-1, 0]:
        print_args('test', opt=args, logger=logger)
        print_device(args.device, batch_size=model_config.total_bs, logger=logger)

    # batch size
    if args.distributed:
        assert model_config.total_bs % world_size == 0
        data_config.dataset.train.others.train_bs = model_config.total_bs // world_size
    else:
        data_config.dataset.train.others.train_bs = model_config.total_bs

    # set random seeds
    if args.seed is not None:
        print_log(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}', logger=logger)
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    # running
    if args.test:
        test(args, data_config, model_config)
    else:
        # define the tensorboard writer
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'val'))
        else:
            train_writer = None
            val_writer = None
        # Train
        train(args, data_config, model_config, train_writer, val_writer, logger=logger)


if __name__ == "__main__":
    main()

