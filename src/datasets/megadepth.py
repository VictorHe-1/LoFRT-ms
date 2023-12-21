import cv2
import os.path as osp
import numpy as np
import mindspore as ms
from mindspore import ops
import logging

logger = logging.getLogger(__name__)

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth
from src.models.utils.supervision_numpy import spvs_coarse


def _interpolate_4d_mask(mask, scale_factor):
    """
    Resize a 4D numpy array using bilinear interpolation in the H and W dimensions.
    :param pred: A 4D numpy array of shape (N, C, H, W).
    :param scale_factor: An integer value representing the scale factor applied to H and W.
    :return: A 4D numpy array of shape (N, C, H', W'), where H' and W' are scaled versions of H and W.
    """
    B, N, H, W = mask.shape
    scaled_H = int(H * scale_factor)
    scaled_W = int(W * scale_factor)

    # warning: C*N should not exceed 512
    # according to: https://stackoverflow.com/a/65160547/6380135
    mask_3d = np.transpose(mask, axes=(2, 3, 1, 0)).reshape((H, W, N * B))

    resize_3d = cv2.resize(mask_3d, (scaled_W, scaled_H), interpolation=cv2.INTER_NEAREST)

    resized_mask = np.transpose(resize_3d.reshape((scaled_H, scaled_W, N, B)), axes=(3, 2, 0, 1))

    return resized_mask


class MegaDepthDataset:
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 config=None,
                 output_idx=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding and depth_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        self.config = config
        self.output_idx = output_idx  # [0, 2, 15, 16, 8, 9, 17, 18, 19]
        # ms special
        self.init_output_columns()


    def init_output_columns(self):
        self.output_columns = [
            'image0',
            'depth0',
            'image1',
            'depth1',
            'T_0to1',
            'T_1to0',
            'K0',
            'K1',
            'scale0',
            'scale1',
            'dataset_name',
            'scene_id',
            'pair_id',
            'pair_names_0',
            'pair_names_1',
            'mask0',
            'mask1']
        # if self.mode in ['train', 'val']:
        #     self.output_columns += ['mask0', 'mask1']

    def get_output_columns(self):
        return self.output_columns

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])

        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        # read depth. shape: (h, w)
        if self.mode in ['train', 'val']:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]), pad_to=self.depth_max_size)
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]), pad_to=self.depth_max_size)
        else:
            depth0 = depth1 = []
        # read intrinsics of original size
        K_0 = np.array(self.scene_info['intrinsics'][idx0].copy()).reshape(3, 3)
        K_1 = np.array(self.scene_info['intrinsics'][idx1].copy()).reshape(3, 3)

        # read and compute relative poses
        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = np.matmul(T1, np.linalg.inv(T0))[:4, :4]  # (4, 4)
        T_1to0 = np.linalg.inv(T_0to1)

        data = [image0, depth0, image1, depth1, T_0to1, T_1to0, K_0, K_1, scale0, scale1,
                'MegaDepth', self.scene_id, idx,
                self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]]
        # for LoFTR training
        if mask0 is not None:  # img_padding is True
            stacked_mask = np.stack([mask0, mask1], axis=0)[None].astype(np.int32)
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = _interpolate_4d_mask(stacked_mask,
                                                             scale_factor=self.coarse_scale)[0].astype(bool)
            data.extend([ts_mask_0, ts_mask_1])
        for idx, item in enumerate(data):
            if not isinstance(item, np.ndarray):
                data[idx] = np.array(item)
            if data[idx].dtype == np.float64:
                data[idx] = data[idx].astype(np.float32)
        if self.mode == 'train':
            data, self.output_columns = spvs_coarse(data, self.config, self.output_columns)
            data = [data[idx] for idx in self.output_idx]
        return tuple(data)
