import torch.utils.data as data
import numpy as np
import os
import sys
import Datasets.dataTransforms as dataTransforms
from .readData import readData
import random
import json
from .datasetBuilder import DATASETS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py

@DATASETS.register_module()
class PCN(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.in_npoints = config.in_npoints
        self.subset = config.subset

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return dataTransforms.Compose([
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': self.in_npoints
                    },
                    'objects': ['partial']
                },
                {
                    'callback': 'RandomMirrorPoints',
                    'objects': ['partial', 'gt']
                },
                # {
                #     'callback': 'RandomRotatePoints',
                #     'objects': ['partial', 'gt']
                # },
                # {
                #     'callback': 'RandomScalePoints',
                #     'parameters': {
                #     'scale': 1.2
                #     },
                #     'objects': ['partial', 'gt']
                # },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }
            ])
        else:
            return dataTransforms.Compose([
                {
                    'callback': 'SamplePoints',
                    'parameters': {
                        'n_points': self.in_npoints
                    },
                    'objects': ['partial']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }
            ])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            # print('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (dc['taxonomy_id'], s),
                })

        # print(f'Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = readData.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)
