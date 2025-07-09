import torch
import torch.utils.data as tud
from utils import registry
from utils.misc import *


DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args = None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT):
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return DATASETS.build(cfg, default_args=default_args)


def dataset_builder(args, dataset_config):
    # dataset = build_dataset_from_cfg(config._base_, config.others)
    dataset = build_dataset_from_cfg(dataset_config, dataset_config.others)
    shuffle = dataset_config.others.subset == 'train'
    if args.distributed:
        sampler = tud.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = tud.DataLoader(dataset, batch_size=dataset_config.others.train_bs if shuffle else 1,
                                    num_workers=int(args.num_workers),
                                    drop_last=dataset_config.others.subset == 'train',
                                    worker_init_fn=worker_init_fn,
                                    sampler=sampler)
    else:
        sampler = None
        dataloader = tud.DataLoader(dataset, batch_size=dataset_config.others.train_bs if shuffle else 1,
                                    shuffle=shuffle,
                                    drop_last=dataset_config.others.subset == 'train',
                                    num_workers=int(args.num_workers),
                                    worker_init_fn=worker_init_fn)
    return sampler, dataloader
