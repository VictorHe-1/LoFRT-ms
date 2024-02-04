import os
import math
import bisect
import logging
from os import path as osp
from collections import abc

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from mindspore.dataset import (
    DistributedSampler,
    RandomSampler
)
from mindspore.communication import get_group_size, get_rank

from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.datasets.megadepth import MegaDepthDataset
from src.datasets.scannet import ScanNetDataset
from src.datasets.numpy_sampler import RandomConcatSampler

logger = logging.getLogger(__name__)

'''
Note that: we don't use RandomConcatSampler here
because the RandomConcatSampler samples a subset of a ConcatDataset every epoch.
Here we use the full dataset for every epoch.
'''


class ConcatDataset:
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        if len(datasets) == 0:
            raise ValueError("datasets passed to ConcatDataset cannot be empty!")
        self.datasets = datasets
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.sampler = None
        self.output_columns = datasets[0].get_output_columns()

    def get_output_columns(self):
        return self.output_columns

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class MultiSceneDataModule:
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """

    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.trainval_data_source = config.DATASET.TRAINVAL_DATA_SOURCE
        self.test_data_source = config.DATASET.TEST_DATA_SOURCE
        # training and validating
        self.train_data_root = config.DATASET.TRAIN_DATA_ROOT
        self.train_pose_root = config.DATASET.TRAIN_POSE_ROOT  # (optional)
        self.train_npz_root = config.DATASET.TRAIN_NPZ_ROOT
        self.train_list_path = config.DATASET.TRAIN_LIST_PATH
        self.train_intrinsic_path = config.DATASET.TRAIN_INTRINSIC_PATH
        self.val_data_root = config.DATASET.VAL_DATA_ROOT
        self.val_pose_root = config.DATASET.VAL_POSE_ROOT  # (optional)
        self.val_npz_root = config.DATASET.VAL_NPZ_ROOT
        self.val_list_path = config.DATASET.VAL_LIST_PATH
        self.val_intrinsic_path = config.DATASET.VAL_INTRINSIC_PATH
        # testing
        self.test_data_root = config.DATASET.TEST_DATA_ROOT
        self.test_pose_root = config.DATASET.TEST_POSE_ROOT  # (optional)
        self.test_npz_root = config.DATASET.TEST_NPZ_ROOT
        self.test_list_path = config.DATASET.TEST_LIST_PATH
        self.test_intrinsic_path = config.DATASET.TEST_INTRINSIC_PATH

        # 2. dataset config
        # general options
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        # self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']
        self.augment_fn = None

        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD  # True
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD  # True
        self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
        self.coarse_scale = 1 / config.LOFTR.RESOLUTION[0]  # 0.125. for training loftr.

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,  # 1
            'num_workers': config.TRAINER.NUM_WORKERS
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': config.TRAINER.NUM_WORKERS
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': config.TRAINER.NUM_WORKERS
        }

        # 4. sampler
        self.data_sampler = config.TRAINER.DATA_SAMPLER
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT
        self.sampler = None

        # (optional) RandomSampler for debugging

        # misc configurations
        self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66
        self.stage = None
        self.config = config
        self.train_pad_num_gt_min = config.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN

    def setup(self, stage=None, distribute=False, output_idx=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"
        self.stage = stage
        if distribute:
            self.world_size = get_group_size()
            self.rank = get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        else:
            self.world_size = 1
            self.rank = 0
            logger.warning("set world_size=1 and rank=0")

        if stage == 'fit':
            self.train_dataset = self._setup_dataset(
                self.train_data_root,
                self.train_npz_root,
                self.train_list_path,
                self.train_intrinsic_path,
                mode='train',
                min_overlap_score=self.min_overlap_score_train,
                pose_dir=self.train_pose_root,
                output_idx=output_idx
            )
            self.sampler = RandomConcatSampler(self.train_dataset,
                                               self.n_samples_per_subset,
                                               self.subset_replacement,
                                               self.shuffle,
                                               self.repeat,
                                               self.seed)
            # setup multiple (optional) validation subsets
            if isinstance(self.val_list_path, (list, tuple)):
                self.val_dataset = []
                if not isinstance(self.val_npz_root, (list, tuple)):
                    self.val_npz_root = [self.val_npz_root for _ in range(len(self.val_list_path))]
                for npz_list, npz_root in zip(self.val_list_path, self.val_npz_root):
                    self.val_dataset.append(self._setup_dataset(
                        self.val_data_root,
                        npz_root,
                        npz_list,
                        self.val_intrinsic_path,
                        mode='val',
                        min_overlap_score=self.min_overlap_score_test,
                        pose_dir=self.val_pose_root))
            else:
                self.val_dataset = self._setup_dataset(
                    self.val_data_root,
                    self.val_npz_root,
                    self.val_list_path,
                    self.val_intrinsic_path,
                    mode='val',
                    min_overlap_score=self.min_overlap_score_test,
                    pose_dir=self.val_pose_root)
            logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')
        else:  # stage == 'test
            self.test_dataset = self._setup_dataset(
                self.test_data_root,
                self.test_npz_root,
                self.test_list_path,
                self.test_intrinsic_path,
                mode='test',
                min_overlap_score=self.min_overlap_score_test,
                pose_dir=self.test_pose_root)
            logger.info(f'[rank:{self.rank}]: Test Dataset loaded!')

    def _setup_dataset(self,
                       data_root,
                       split_npz_root,
                       scene_list_path,
                       intri_path,
                       mode='train',
                       min_overlap_score=0.,
                       pose_dir=None,
                       output_idx=None):
        """ Setup train / val / test set"""
        with open(scene_list_path, 'r') as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        if mode == 'train':
            local_npz_names = get_local_split(npz_names, self.world_size, self.rank, self.seed)
        else:
            local_npz_names = npz_names
        logger.info(f'[rank {self.rank}]: {len(local_npz_names)} scene(s) assigned.')

        dataset_builder = self._build_concat_dataset_parallel \
            if self.parallel_load_data \
            else self._build_concat_dataset
        return dataset_builder(data_root, local_npz_names, split_npz_root, intri_path,
                               mode=mode, min_overlap_score=min_overlap_score, pose_dir=pose_dir, output_idx=output_idx)

    def _build_concat_dataset(
            self,
            data_root,
            npz_names,
            npz_dir,
            intrinsic_path,
            mode,
            min_overlap_score=0.,
            pose_dir=None,
            output_idx=None,
    ):
        datasets = []
        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
        if str(data_source).lower() == 'megadepth':
            npz_names = [f'{n}.npz' for n in npz_names]
        for npz_name in tqdm(npz_names,
                             desc=f'[rank:{self.rank}] loading {mode} datasets',
                             disable=int(self.rank) != 0):
            # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
            npz_path = osp.join(npz_dir, npz_name)
            if data_source == 'ScanNet':
                datasets.append(
                    ScanNetDataset(data_root,
                                   npz_path,
                                   intrinsic_path,
                                   mode=mode,
                                   min_overlap_score=min_overlap_score,
                                   augment_fn=augment_fn,
                                   pose_dir=pose_dir))
            elif data_source == 'MegaDepth':
                datasets.append(
                    MegaDepthDataset(data_root,
                                     npz_path,
                                     mode=mode,
                                     min_overlap_score=min_overlap_score,
                                     img_resize=self.mgdpt_img_resize,
                                     df=self.mgdpt_df,
                                     img_padding=self.mgdpt_img_pad,
                                     depth_padding=self.mgdpt_depth_pad,
                                     augment_fn=augment_fn,
                                     coarse_scale=self.coarse_scale,
                                     config=self.config,
                                     output_idx=output_idx,
                                     train_pad_num_gt_min=self.train_pad_num_gt_min))
            else:
                raise NotImplementedError()
        return ConcatDataset(datasets)  # TODO: convert each item into GeneratorDataset and concat them.

    def _build_concat_dataset_parallel(
            self,
            data_root,
            npz_names,
            npz_dir,
            intrinsic_path,
            mode,
            min_overlap_score=0.,
            pose_dir=None,
    ):
        augment_fn = self.augment_fn if mode == 'train' else None
        data_source = self.trainval_data_source if mode in ['train', 'val'] else self.test_data_source
        if str(data_source).lower() == 'megadepth':
            npz_names = [f'{n}.npz' for n in npz_names]
        with tqdm_joblib(tqdm(desc=f'[rank:{self.rank}] loading {mode} datasets',
                              total=len(npz_names), disable=int(self.rank) != 0)):
            if data_source == 'ScanNet':
                datasets = Parallel(n_jobs=math.floor(len(os.sched_getaffinity(0)) * 0.9 / get_group_size()))(
                    delayed(lambda x: _build_dataset(
                        ScanNetDataset,
                        data_root,
                        osp.join(npz_dir, x),
                        intrinsic_path,
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        augment_fn=augment_fn,
                        pose_dir=pose_dir))(name)
                    for name in npz_names)
            elif data_source == 'MegaDepth':
                # TODO: _pickle.PicklingError: Could not pickle the task to send it to the workers.
                raise NotImplementedError()
                datasets = Parallel(n_jobs=math.floor(len(os.sched_getaffinity(0)) * 0.9 / get_group_size()))(
                    delayed(lambda x: _build_dataset(
                        MegaDepthDataset,
                        data_root,
                        osp.join(npz_dir, x),
                        mode=mode,
                        min_overlap_score=min_overlap_score,
                        img_resize=self.mgdpt_img_resize,
                        df=self.mgdpt_df,
                        img_padding=self.mgdpt_img_pad,
                        depth_padding=self.mgdpt_depth_pad,
                        augment_fn=augment_fn,
                        coarse_scale=self.coarse_scale))(name)
                    for name in npz_names)
            else:
                raise ValueError(f'Unknown dataset: {data_source}')
        return ConcatDataset(datasets)


def _build_dataset(dataset, *args, **kwargs):
    return dataset(*args, **kwargs)
