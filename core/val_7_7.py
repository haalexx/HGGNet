from tracemalloc import is_tracing
import torch
import json
from pathlib import Path
import dataLoaders
import utils.misc as misc
from tqdm import tqdm
from utils.config import *
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from utils.general import increment_path, create_experiment_dir
from utils import parser
from models import HGGNet
from tensorboardX import SummaryWriter
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import utils.dist_utils as dist_utils
import logging


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, data_config, model_config, logger=None):
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['CoarseLossL1', 'P1LossL1', 'P2LossL1', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    s = 'validating'
    n_samples = len(test_dataloader)
    # pbar = tqdm(test_dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            dataset_name = data_config.dataset.val.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                npoints = data_config.dataset.val.N_POINTS
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                                                      fixed_points=None)
                partial = partial.cuda()
            elif dataset_name == 'C3D':
                partial = data[0].cuda()
                gt = data[1].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            coarse_points, p1, p2, dense_points = base_model(partial)

            coarse_loss_l1 = ChamferDisL1(coarse_points, gt)
            p1_loss_l1 = ChamferDisL1(p1, gt)
            p2_loss_l1 = ChamferDisL1(p2, gt)
            dense_loss_l1 = ChamferDisL1(dense_points, gt)
            dense_loss_l2 = ChamferDisL2(dense_points, gt)

            if args.distributed:
                coarse_loss_l1 = dist_utils.reduce_tensor(coarse_loss_l1, args)
                p1_loss_l1 = dist_utils.reduce_tensor(p1_loss_l1, args)
                p2_loss_l1 = dist_utils.reduce_tensor(p2_loss_l1, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([coarse_loss_l1.item() * 1000, p1_loss_l1.item() * 1000, p2_loss_l1.item() * 1000,
                                dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt)

            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)

            # if taxonomy_id not in category_metrics:
            #     category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            # category_metrics[taxonomy_id].update(_metrics)

            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input' % idx, input_pc, epoch, dataformats='HWC')

            #     sparse = coarse_points.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')

            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')

            # Print results
            if args.local_rank in [-1, 0] and (idx + 1) % 20 == 0:
                print('Test[%d/%d]: Losses = %s Metrics = %s' %
                    (idx + 1, n_samples, ['%.2f' % l for l in test_losses.val()],
                    ['%.4f' % m for m in _metrics]))

        for _, v in category_metrics.items():
            test_metrics.update(v.avg())

        if args.distributed:
            torch.cuda.synchronize()
    print_log('[Validation] EPOCH: %d  Metrics (F-Score, CDL1, CDL2) = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)
    # Print testing results
    shapenet_dict = json.load(open('./Datasets/shapenet_synset_dict.json', 'r'))
    print_log('-------------------- TEST RESULTS ---------------------', logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(3), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(model_config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1 / 4,
    'median': 1 / 2,
    'hard': 3 / 43
}


if __name__ == "__main__":
    args = parser.get_args()
    data_config, model_config = get_config(args)
    args.experiment_path = str(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))
    args.tfboard_path = os.path.join(args.experiment_path, 'TFBoard')
    args.log_name = args.name
    create_experiment_dir(args)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    model = HGGNet(model_config.model).cuda()

    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'val'))
        else:
            train_writer = None
            val_writer = None

    if args.launcher == 'none':
        args.distributed = False
    else:
        # DDP mode
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    print(f"{colorstr('Distributed training:')} {args.distributed}")

    _, test_dataloader = dataLoaders.dataset_builder(args, data_config.dataset.val)
    metrics = validate(model, test_dataloader, 1, ChamferDisL1, ChamferDisL2, val_writer, args, data_config,
                       model_config)
