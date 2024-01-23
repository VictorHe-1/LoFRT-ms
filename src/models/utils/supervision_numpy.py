import logging
from math import log

import numpy as np

from .geometry_numpy import warp_kpts

_logger = logging.getLogger(__name__)


##############  ↓  Coarse-Level supervision  ↓  ##############

def repeat(tensor, pattern, c):
    tensor = tensor.reshape(tensor.shape[0], -1)
    tensor = np.expand_dims(tensor, axis=-1)
    tensor = np.tile(tensor, (1, 1, c))
    return tensor


def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.astype(bool)] = 0
    return grid_pt


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: bool = True
):
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])
    """
    xs = np.linspace(0, width - 1, width)
    ys = np.linspace(0, height - 1, height)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)  # WxHx2
    return np.expand_dims(base_grid.transpose(1, 0, 2), axis=0)  # 1xHxWx2


def spvs_coarse(data, config, data_cols, train_pad_num_gt_min):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    N, H0, W0 = data[0].shape
    _, H1, W1 = data[2].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data[8][None][:, None]  # if 'scale0' in data_cols else scale
    scale1 = scale * data[9][None][:, None]  # if 'scale1' in data_cols else scale

    h0 = H0 // scale
    w0 = W0 // scale
    h1 = H1 // scale
    w1 = W1 // scale

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = np.tile(create_meshgrid(h0, w0, False).reshape(1, h0 * w0, 2), (N, 1, 1))  # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = np.tile(create_meshgrid(h1, w1, False).reshape(1, h1 * w1, 2), (N, 1, 1))
    grid_pt1_i = scale1 * grid_pt1_c
    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data[15][None])  # 15
    grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data[16][None])  # 16
    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(grid_pt0_i,
                             data[1][None],
                             data[3][None],
                             data[4][None],
                             data[6][None],
                             data[7][None])
    _, w_pt1_i = warp_kpts(grid_pt1_i,
                             data[3][None],
                             data[1][None],
                             data[5][None],
                             data[7][None],
                             data[6][None])
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0
    # 3. check if mutual nearest neighbor
    w_pt0_c_round = np.round(w_pt0_c[:, :, :]).astype(np.int64)
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = np.round(w_pt1_c[:, :, :]).astype(np.int64)
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        con1 = (pt[..., 0] < 0).astype(np.int16)
        con2 = (pt[..., 0] >= w).astype(np.int16)
        con3 = (pt[..., 1] < 0).astype(np.int16)
        con4 = (pt[..., 1] >= h).astype(np.int16)
        return (con1 | con2 | con3 | con4).astype(bool)

    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = np.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], axis=0)
    correct_0to1 = loop_back == np.tile(np.arange(h0 * w0)[None], (N, 1))
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = np.zeros((N, h0 * w0, h1 * w1))
    b_ids, i_ids = np.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.append(conf_matrix_gt[0].astype(np.float32))
    data_cols.append("conf_matrix_gt")

    # 5. save coarse matches(gt) for training fine level
    # 'spv_b_ids', 'spv_i_ids', 'spv_j_ids',
    data_cols.extend(['spv_w_pt0_i', 'spv_pt1_i', 'spv_i_ids', 'spv_j_ids'])
    data.extend([w_pt0_i[0].astype(np.float32), grid_pt1_i[0].astype(np.float32),
                 i_ids[:train_pad_num_gt_min].astype(np.int32), j_ids[:train_pad_num_gt_min].astype(np.int32)])
    return data, data_cols

##############   Fine-Level supervision is moved to the loftr model  ##############
