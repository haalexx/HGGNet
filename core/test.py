import Datasets
import json
from tqdm import tqdm
import torch
from models import HGGNet
from utils import misc
from utils.logger import *
from utils.metrics import Metrics
from utils.build_utils import load_model
from utils.AverageMeter import AverageMeter
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.infer_utils import savePointCloud, create_output_dir


def test(args, data_config, model_config):
    LOGGER.info(f"{colorstr('Tester start ...')}")
    _, test_dataloader = Datasets.dataset_builder(args, data_config.dataset.test)
    model = HGGNet(model_config.model)
    load_model(model, args.weights)
    model.cuda()

    # DDP
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    model.eval()
    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)  # bs is 1
    s = 'testing......'
    dataset_name = data_config.dataset.test.NAME
    output_path = os.path.join(args.experiment_path, 'testOutput')
    create_output_dir(output_path, dataset_name=dataset_name)
    best_model = dict()
    best_value = dict()

    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for idx, (taxonomy_ids, model_ids, data) in enumerate(pbar):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)

                test_losses.update(
                    [sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                     dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt)
                test_metrics.update(_metrics)
                file_name = model_id + '.pcd'
                file_path = os.path.join(output_path, taxonomy_id, file_name)
                savePointCloud(file_path, dense_points.squeeze(0))
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                current_metrics = Metrics('CDL1', _metrics)
                if taxonomy_id not in best_model:
                    best_model[taxonomy_id] = model_id
                    best_value[taxonomy_id] = current_metrics
                if not best_value[taxonomy_id].better_than(current_metrics):
                    best_model[taxonomy_id] = model_id
                    best_value[taxonomy_id] = current_metrics

            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            elif dataset_name == 'C3D':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)

                test_losses.update(
                    [sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000,
                     dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt)
                test_metrics.update(_metrics)
                file_name = model_id + '.pcd'
                file_path = os.path.join(output_path, taxonomy_id, file_name)
                savePointCloud(file_path, dense_points.squeeze(0))
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

                current_metrics = Metrics('CDL1', _metrics)
                if taxonomy_id not in best_model:
                    best_model[taxonomy_id] = model_id
                    best_value[taxonomy_id] = current_metrics
                if not best_value[taxonomy_id].better_than(current_metrics):
                    best_model[taxonomy_id] = model_id
                    best_value[taxonomy_id] = current_metrics
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

        if dataset_name == 'KITTI':
            return
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        LOGGER.info('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]))

    # Print testing results
    shapenet_dict = json.load(open('./Datasets/shapenet_synset_dict.json', 'r'))
    LOGGER.info('============================ TEST RESULTS ============================')
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    LOGGER.info(msg)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        LOGGER.info(msg)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    LOGGER.info(msg)

    LOGGER.info('================ Best Model ===================')
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample'
    LOGGER.info(msg)

    for taxonomy_id in best_model:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += best_model[taxonomy_id]
        LOGGER.info(msg)

    return


