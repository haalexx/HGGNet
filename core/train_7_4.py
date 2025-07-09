import torch
import torch.nn as nn
import time
import dataLoaders
import utils.dist_utils as dist_utils
from models import HGGNet
from utils import misc
from utils.general import check_suffix, intersect_dicts
from utils.logger import *
from utils.metrics import Metrics
import logging
from utils.build_utils import build_opti_sche, resume_model, resume_optimizer, load_model, save_checkpoint
from utils.AverageMeter import AverageMeter
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from core.val_7_4 import validate


def train(args, data_config, model_config, train_writer, val_writer):
    weights = args.weights
    # build dataset
    train_sampler, train_dataloader = dataLoaders.dataset_builder(args, data_config.dataset.train)
    _, test_dataloader = dataLoaders.dataset_builder(args, data_config.dataset.val)

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None
    alpha = 0.9

    # Model
    check_suffix(weights, '.pth')  # check weights
    pretrained = weights.endswith('.pth')
    if pretrained:
        ckpt = torch.load(weights)  # load checkpoint
        model = HGGNet(model_config.model)  # create
        csd = ckpt['base_model']  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict())  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.cuda()                # to cuda
        logging.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    elif args.resume:    # resume ckpts
        model = HGGNet(model_config.model)  # create
        start_epoch, best_metrics = resume_model(model, args.resume_weights, args)
        model.cuda()                # to cuda
        best_metrics = Metrics(model_config.consider_metric, best_metrics)
    else:
        model = HGGNet(model_config.model).cuda()  # create

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logging.info('Using Synchronized BatchNorm ...')
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % torch.cuda.device_count()],
                                                    find_unused_parameters=True)
        logging.info(f"{colorstr('Using Distributed Data parallel ...')}")
    else:
        logging.info(f"{colorstr('Using Data parallel ...')}")
        model = nn.DataParallel(model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = build_opti_sche(model, model_config)

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    if args.resume:
        resume_optimizer(optimizer, args.resume_weights)

    # Train-Val
    # training
    model.zero_grad()
    for epoch in range(start_epoch, model_config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        logging.info('========================= Epoach: %d ==============================='% epoch)

        num_iter = 0

        model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        # pbar = enumerate(train_dataloader)
        # LOGGER.info(('\n' + '%10s' + '%15s' * 4) % ('Epoch', 'gpu_mem', 'SparseLoss', 'DenseLoss', 'lr'))
        # if args.rank in [-1, 0]:
        #     pbar = tqdm(pbar, total=n_batches, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            dataset_name = data_config.dataset.train.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'C3D':
                partial = data[0].cuda()
                gt = data[1].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
            try:
                ret = model(partial)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            # sparse_loss, dense_loss, tran_loss = model.module.get_loss(ret, gt, partial)
            denoise_loss, recon_loss = model.module.get_loss(ret, gt)

            # _loss = dense_loss + alpha*sparse_loss + beta*tran_loss
            _loss = recon_loss + denoise_loss
            _loss.backward()

            # forward
            if num_iter == model_config.step_per_update:
                num_iter = 0
                optimizer.step()
                model.zero_grad()

            if args.distributed:
                denoise_loss = dist_utils.reduce_tensor(denoise_loss, args)
                recon_loss = dist_utils.reduce_tensor(recon_loss, args)
                losses.update([denoise_loss.item()*1000, recon_loss.item()*1000])
            else:
                losses.update([denoise_loss.item() * 1000, recon_loss.item() * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/denoise', denoise_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/recon', recon_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if args.rank in [-1, 0] and (idx % 20 == 0):
                print('epoch:%d/%d, data:%d/%d, denoise_loss:%.2f, recon_loss:%.2f, lr:%.8f' % (
                    epoch, model_config.max_epoch, idx, n_batches,
                    losses.val(0), losses.val(1), optimizer.param_groups[0]['lr'])
                    )

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/denoise', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/recon', losses.avg(1), epoch)
        logging.info('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
                    (epoch, epoch_end_time - epoch_start_time, ['%.4f' % loss for loss in losses.avg()]))

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, data_config,
                               model_config)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                save_checkpoint(model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args)
            
            if best_metrics is not None:
                best_dict = best_metrics.state_dict()
                msg = ''
                msg = 'Best Result\t\t\t\t\t'
                for key, value in best_dict.items():
                    msg += '%.3f \t' % value
                logging.info(msg)

        save_checkpoint(model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args)
        if (model_config.max_epoch - epoch) < 3:
            save_checkpoint(model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args)


    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
